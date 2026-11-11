# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Xylolabs
"""TT-NN's block floating point grids in torch, for checks that run without ttnn.

A block is 16 consecutive values along the last dimension of the [in, out] layout TT-NN's matmul
reads (16 output channels of one input channel). The block shares the largest exponent among its
values; each value keeps a sign and a mantissa with an explicit leading one: 7 bits for bfloat8_b,
3 bits for bfloat4_b. bfp() rounds as tt-metal's host packer does (round to nearest even, the
mantissa clamped at its maximum, a zero exponent field flushed to zero). From tensix-bfp-gptq
(ORIGINS.md)."""

import torch

MANTISSA_BITS = {"bfp8": 7, "bfp4": 3}


def bfp(x, mant_bits, block=16):
    """x rounded to the block floating point grid as tt-metal's host packer rounds it, as float32."""
    shape = x.shape
    assert shape[-1] % block == 0, f"bfp: last dimension {shape[-1]} is not a multiple of {block}"
    b = x.float().contiguous().reshape(-1, block).view(torch.int32)
    exp = (b >> 23) & 0xFF
    mant = (b & 0x7FFFFF) | 0x800000
    shared = exp.amax(1, keepdim=True)
    m = mant >> (shared - exp).clamp(max=31)
    shift = 24 - mant_bits
    rnd = m & ((1 << shift) - 1)
    m = m >> shift
    tie = 1 << (shift - 1)
    m = m + ((rnd > tie) | ((rnd == tie) & ((m & 1) == 1))).int()
    m = m.clamp(max=(1 << mant_bits) - 1)
    m = torch.where(exp == 0, torch.zeros_like(m), m)
    v = torch.ldexp(m.float(), (shared - 127 - (mant_bits - 1)).float())
    return torch.where(b < 0, -v, v).reshape(shape)


def on_grid(w, fmt):
    """Whether a PyTorch [out, in] weight already lies on fmt's grid ("bf16", "bfp8" or "bfp4") in
    TT-NN's [in, out] layout, so that packing it changes no value."""
    if fmt == "bf16":
        return torch.equal(w.float().bfloat16().float(), w.float())
    t = w.float().T.contiguous()
    return torch.equal(bfp(t, MANTISSA_BITS[fmt]), t)
