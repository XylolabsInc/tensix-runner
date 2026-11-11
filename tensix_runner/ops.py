# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Xylolabs
"""Small building blocks on ttnn shared by the models: compute configs, matrices, norms and the MLP."""

import torch

from . import weights

TILE = 32


# Wormhole's source registers hold 19 bits, so a matmul that reloads fp32 partial sums into them is
# undefined there (ttsim's Wormhole stops on it); matmuls on Wormhole accumulate in bf16 instead
_ARCH = {"wormhole": False}


def set_arch(device):
    """Record the device's architecture for compute_config (called once per model load)."""
    import ttnn

    _ARCH["wormhole"] = device.arch() == ttnn.device.Arch.WORMHOLE_B0


def compute_config(fidelity="hifi4", matmul=False):
    """A compute config without approximations: fp32 accumulation, except for matmuls on
    Wormhole (see _ARCH)."""
    import ttnn

    f = {
        "lofi": ttnn.MathFidelity.LoFi,
        "hifi2": ttnn.MathFidelity.HiFi2,
        "hifi4": ttnn.MathFidelity.HiFi4,
    }[fidelity]
    fp32 = not (matmul and _ARCH["wormhole"])
    return ttnn.WormholeComputeKernelConfig(
        math_fidelity=f, math_approx_mode=False, fp32_dest_acc_en=fp32, packer_l1_acc=True
    )


def sdpa_config(device, k_chunk=32):
    """SDPA's program config with the exact exponential: ttnn's SDPA ops take the approximate one
    unless a program config says otherwise (math_approx_mode does not reach it). Chunks of 32
    query rows and k_chunk key rows; decode passes its own k_chunk, never 0 (with a program config,
    0 leaves the chunk to the kernels, which then misplace a sliding window shorter than the
    cache: PCC 0.5-0.8 against torch for a window of 32 on ttsim)."""
    import ttnn

    return ttnn.SDPAProgramConfig(
        compute_with_storage_grid_size=device.compute_with_storage_grid_size(),
        q_chunk_size=32,
        k_chunk_size=k_chunk,
        exp_approx_mode=False,
    )


def matrix(device, w, fmt):
    """A PyTorch [out, in] weight on the device as [in, out] at fmt, in DRAM (packed exactly)."""
    import ttnn

    return ttnn.to_device(weights.pack(w, fmt), device, memory_config=ttnn.DRAM_MEMORY_CONFIG)


def norm_weight(device, w, offset=0.0):
    """An RMSNorm weight [dim] in the form ttnn.rms_norm reads: bf16 rows of 32, row major. offset
    1.0 for checkpoints that store the weight zero-centered (Qwen3.5's RMSNorm scales by 1 + w)."""
    import ttnn

    dim = w.shape[-1]
    assert dim % TILE == 0, f"norm width {dim} is not a multiple of {TILE}"
    return ttnn.as_tensor(
        (w.float() + offset).to(torch.bfloat16).reshape(1, 1, dim // TILE, TILE),
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


def rms_norm(x, w, eps, mc=None):
    import ttnn

    return ttnn.rms_norm(
        x,
        epsilon=eps,
        weight=w,
        compute_kernel_config=compute_config("hifi4"),
        memory_config=mc or ttnn.DRAM_MEMORY_CONFIG,
    )


def linear(x, w, fidelity="hifi4", mc=None):
    import ttnn

    return ttnn.linear(
        x,
        w,
        compute_kernel_config=compute_config(fidelity, matmul=True),
        memory_config=mc or ttnn.DRAM_MEMORY_CONFIG,
    )


def fidelity_for(fmt):
    """The matmul fidelity for a weight format: HiFi4 for every format. The fidelity phases split
    the bf16 activations as well as the weights, so a narrow weight format does not make HiFi2
    exact: with HiFi2 on the BFP4 matrices, Gemma 4 E2B's layer outputs in a whole prefill sat
    about twice as far from transformers' float32 ones as with HiFi3 or HiFi4."""
    return "hifi4"


class SwiGLU:
    """gate, up and down: down(silu(gate(x)) * up(x))."""

    def __init__(self, device, gate, up, down, fmts):
        self.fmts = fmts
        self.gate = matrix(device, gate, fmts[0])
        self.up = matrix(device, up, fmts[1])
        self.down = matrix(device, down, fmts[2])

    def __call__(self, x):
        import ttnn

        g = linear(x, self.gate, fidelity_for(self.fmts[0]))
        u = linear(x, self.up, fidelity_for(self.fmts[1]))
        a = self._gated(g, u)
        ttnn.deallocate(g)
        ttnn.deallocate(u)
        y = linear(a, self.down, fidelity_for(self.fmts[2]))
        ttnn.deallocate(a)
        return y

    def _gated(self, g, u):
        import ttnn

        return ttnn.multiply(g, u, input_tensor_a_activations=[ttnn.UnaryOpType.SILU])


class GeluTanhMLP(SwiGLU):
    """down(gelu(gate(x)) * up(x)) with GELU's tanh form (transformers' gelu_pytorch_tanh)."""

    def _gated(self, g, u):
        import ttnn

        a = gelu_tanh(g)
        out = ttnn.multiply(a, u)
        ttnn.deallocate(a)
        return out


def gelu_tanh(x):
    """0.5 x (1 + tanh(sqrt(2 / pi) (x + 0.044715 x^3))), as torch's gelu(approximate="tanh")."""
    import ttnn

    return ttnn.gelu(x, variant=ttnn.GeluVariant.Tanh)


def rows(t, n):
    """t [1, 1, *, C] viewed with n logical rows in its tiles (some ops return a partial tile's
    padding rows as logical ones; a residual add then needs both sides at n rows)."""
    import ttnn

    c = t.shape[-1]
    return ttnn.reshape(t, (1, 1, n, c), (1, 1, -(-n // TILE) * TILE, c))
