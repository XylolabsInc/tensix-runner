# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Xylolabs
"""Qwen3.5 (Qwen3.5-4B, Qwen3.5-9B) on one chip: Gated DeltaNet layers and gated attention layers.

Gated attention: q_proj's rows hold each head's q and its output gate in turn; at load they are
split by whole heads into q and gate matrices (exact). Only the first quarter of each head
(rotary_dim) is rotated, with HF's rotate_half inside it; q's and k's channels are permuted at load
by whole tiles so that those pairs sit where rotate_half over the full head pairs them, and the
other channels get cos 1 and sin 0 (a dot product of q and k is unchanged by permuting both). The
attention output is scaled by sigmoid(gate) before o_proj. Norms scale by 1 + w.

Gated DeltaNet: in_proj_qkv and in_proj_z as one matrix, a depthwise causal conv of 4 taps with
SiLU on q|k|v, then ttnn's chunked Gated Delta Rule (prefill) or its fused recurrent step (decode)
on a per-layer fp32 state, then RMSNorm per value head gated by silu(z), and out_proj. The decay
g = -exp(A_log) softplus(a + dt_bias) and beta = sigmoid(b) are computed on the device.

ttnn's Gated Delta Rule ops run on Blackhole; ttsim's Wormhole stops in both (fp32 into the source
registers, fence instructions), so this family loads on Blackhole only for now."""

import math

import torch

from .. import ops
from ..kv import PagedKV
from ..rope import HfRope, inv_freq
from .qwen3 import TILE, head_for

GDN, ATTN = "linear_attention", "full_attention"


class Config:
    def __init__(self, tc):
        self.dim = tc["hidden_size"]
        self.layers = tc["num_hidden_layers"]
        self.types = tc["layer_types"]
        self.heads = tc["num_attention_heads"]
        self.kv_heads = tc["num_key_value_heads"]
        self.head_dim = tc["head_dim"]
        rp = tc.get("rope_parameters") or {}
        self.theta = rp.get("rope_theta", tc.get("rope_theta", 1e7))
        self.rotary = int(self.head_dim * rp.get("partial_rotary_factor", 1.0))
        self.eps = tc["rms_norm_eps"]
        self.vocab = tc["vocab_size"]
        self.k_heads = tc["linear_num_key_heads"]
        self.v_heads = tc["linear_num_value_heads"]
        self.k_dim = tc["linear_key_head_dim"]
        self.v_dim = tc["linear_value_head_dim"]
        self.conv = tc["linear_conv_kernel_dim"]
        assert self.k_dim == self.v_dim, "ttnn's flat Gated Delta Rule takes K == V"
        assert self.rotary % (2 * TILE) == 0 and self.head_dim % (2 * TILE) == 0, (
            "the rotary permutation moves whole tiles"
        )


def head_perm(d, r):
    """The new order of a head's d channels (new index -> old) that puts each rotated pair
    (j, j + r/2), j < r/2, at (j, j + d/2), where rotate_half over all d channels pairs them."""
    h, half = r // 2, d // 2
    return [
        *range(h),
        *range(r, r + half - h),
        *range(h, r),
        *range(r + half - h, d),
    ]


def permute_heads(w, d, perm):
    """Rows of a [heads * d, in] projection reordered within each head by perm."""
    n = w.shape[0] // d
    return w.reshape(n, d, -1)[:, perm].reshape(n * d, -1).contiguous()


def tables(head_dim, rotary, theta, max_seq):
    """cos and sin [max_seq, head_dim] float32 in the permuted channel order (head_perm): the
    rotated pairs at (j, j + head_dim/2) for j < rotary/2, cos 1 and sin 0 elsewhere."""
    h = rotary // 2
    f = inv_freq(head_dim, theta, rotary)  # rotary/2 frequencies, as transformers computes them
    ang = torch.outer(torch.arange(max_seq, dtype=torch.int64).float(), f)
    cos = torch.ones(max_seq, head_dim)
    sin = torch.zeros(max_seq, head_dim)
    for base in (0, head_dim // 2):
        cos[:, base : base + h] = ang.cos()
        sin[:, base : base + h] = ang.sin()
    return cos, sin


class Qwen35Rope(HfRope):
    """HfRope over tables(): the device copies prefill slices and decode gathers rows from."""

    def __init__(self, device, c, max_seq):
        import ttnn

        d = c.head_dim
        cos, sin = tables(d, c.rotary, c.theta, max_seq)
        self.device, self.head_dim, self.max_seq = device, d, max_seq
        dram = ttnn.DRAM_MEMORY_CONFIG

        def put(t, layout, shape):
            return ttnn.as_tensor(
                t.to(torch.bfloat16).reshape(shape),
                dtype=ttnn.bfloat16,
                layout=layout,
                device=device,
                memory_config=dram,
            )

        self.cos_tile = put(cos, ttnn.TILE_LAYOUT, (1, 1, max_seq, d))
        self.sin_tile = put(sin, ttnn.TILE_LAYOUT, (1, 1, max_seq, d))
        self.cos_rows = put(cos, ttnn.ROW_MAJOR_LAYOUT, (max_seq, d))
        self.sin_rows = put(sin, ttnn.ROW_MAJOR_LAYOUT, (max_seq, d))


class GatedAttention:
    def __init__(self, model, i, prefix=None, fmt=None):
        """Layer i's attention, or with prefix (e.g. "mtp.layers.0.") another one whose matrices
        quant4.json does not list, packed at fmt."""
        ex, dev, c = model.export, model.device, model.cfg
        self.model = model
        p = (prefix or f"{ex.prefix}layers.{i}.") + "self_attn."
        if fmt is None:
            fmts = {ex.formats[(i, f"self_attn.{n}_proj")] for n in "qkv"}
            assert len(fmts) == 1, f"layer {i}: q, k and v in different formats {fmts}"
            self.qkv_fmt, self.o_fmt = fmts.pop(), ex.formats[(i, "self_attn.o_proj")]
        else:
            self.qkv_fmt = self.o_fmt = fmt
        d, perm = c.head_dim, model.perm
        qg = ex.get(p + "q_proj.weight").reshape(c.heads, 2, d, -1)
        q = permute_heads(qg[:, 0].reshape(c.heads * d, -1), d, perm)
        gate = qg[:, 1].reshape(c.heads * d, -1).contiguous()
        k = permute_heads(ex.get(p + "k_proj.weight"), d, perm)
        self.wqkv = ops.matrix(dev, torch.cat([q, k, ex.get(p + "v_proj.weight")], 0), self.qkv_fmt)
        self.wgate = ops.matrix(dev, gate, self.qkv_fmt)
        self.wo = ops.matrix(dev, ex.get(p + "o_proj.weight"), self.o_fmt)
        self.q_norm = ops.norm_weight(dev, ex.get(p + "q_norm.weight")[perm], 1.0)
        self.k_norm = ops.norm_weight(dev, ex.get(p + "k_norm.weight")[perm], 1.0)
        self.kv = PagedKV(dev, c.kv_heads, d, model.max_seq)

    def _head_norm(self, x, w, decode):
        import ttnn

        eps = self.model.cfg.eps
        if not decode:
            return ops.rms_norm(x, w, eps)
        mc = x.memory_config()
        y = ops.rms_norm(
            ttnn.to_memory_config(x, ttnn.L1_MEMORY_CONFIG), w, eps, ttnn.L1_MEMORY_CONFIG
        )
        return ttnn.to_memory_config(y, mc)

    def _gated(self, o, h):
        """o * sigmoid(gate(h)), then o_proj."""
        import ttnn

        g = ops.linear(h, self.wgate, ops.fidelity_for(self.qkv_fmt))
        y = ttnn.multiply(g, o, input_tensor_a_activations=[ttnn.UnaryOpType.SIGMOID])
        ttnn.deallocate(g)
        out = ops.linear(y, self.wo, ops.fidelity_for(self.o_fmt))
        ttnn.deallocate(y)
        return out

    def prefill(self, h, cos, sin, n):
        import ttnn

        c = self.model.cfg
        qkv = ops.linear(h, self.wqkv, ops.fidelity_for(self.qkv_fmt))
        q, k, v = ttnn.experimental.nlp_create_qkv_heads(
            qkv,
            num_heads=c.heads,
            num_kv_heads=c.kv_heads,
            transpose_k_heads=False,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(qkv)
        q = ttnn.experimental.rotary_embedding_hf(
            self._head_norm(q, self.q_norm, False), cos, sin, is_decode_mode=False
        )
        k = ttnn.experimental.rotary_embedding_hf(
            self._head_norm(k, self.k_norm, False), cos, sin, is_decode_mode=False
        )
        self.kv.fill(k, v)
        a = ttnn.transformer.scaled_dot_product_attention(
            q,
            k,
            v,
            is_causal=True,
            scale=1.0 / math.sqrt(c.head_dim),
            compute_kernel_config=ops.compute_config("hifi4"),
            program_config=ops.sdpa_config(self.model.device),
        )
        for t in (q, k, v):
            ttnn.deallocate(t)
        o = ttnn.experimental.nlp_concat_heads(a, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(a)
        return self._gated(o, h)

    def decode(self, h, cos, sin, pos, mcs):
        import ttnn

        c = self.model.cfg
        qkv = ops.linear(h, self.wqkv, ops.fidelity_for(self.qkv_fmt), ttnn.L1_MEMORY_CONFIG)
        users = mcs["users"]  # one decode row, or a verify's T rows as T users
        qkv = ttnn.reshape(qkv, (1, 1, users, qkv.shape[-1]), (1, 1, TILE, qkv.shape[-1]))
        q, k, v = ttnn.experimental.nlp_create_qkv_heads_decode(
            qkv, num_heads=c.heads, num_kv_heads=c.kv_heads, memory_config=mcs["heads"]
        )
        ttnn.deallocate(qkv)
        q = ttnn.experimental.rotary_embedding_hf(
            self._head_norm(q, self.q_norm, True), cos, sin, is_decode_mode=True
        )
        k = ttnn.experimental.rotary_embedding_hf(
            self._head_norm(k, self.k_norm, True), cos, sin, is_decode_mode=True
        )
        self.kv.update(k, v, pos, users)
        ttnn.deallocate(k)
        ttnn.deallocate(v)
        a = self.kv.attend(q, pos, 1.0 / math.sqrt(c.head_dim), ops.compute_config("hifi4"), users)
        ttnn.deallocate(q)
        a = ttnn.to_memory_config(a, mcs["sdpa_out"])
        o = ttnn.experimental.nlp_concat_heads_decode(a, num_heads=c.heads)
        ttnn.deallocate(a)
        o = ops.rows(ttnn.to_memory_config(o, ttnn.DRAM_MEMORY_CONFIG), users)
        return self._gated(o, h)


class GatedDeltaNet:
    def __init__(self, model, i):
        import ttnn

        ex, dev, c = model.export, model.device, model.cfg
        self.model = model
        p = f"{ex.prefix}layers.{i}.linear_attn."
        f = ex.formats
        self.qkv_fmt, self.z_fmt = (
            f[(i, "linear_attn.in_proj_qkv")],
            f[(i, "linear_attn.in_proj_z")],
        )
        self.ab_fmt = f[(i, "linear_attn.in_proj_a")]
        assert f[(i, "linear_attn.in_proj_b")] == self.ab_fmt, f"layer {i}: a and b differ"
        self.out_fmt = f[(i, "linear_attn.out_proj")]
        self.kd, self.vd = c.k_heads * c.k_dim, c.v_heads * c.v_dim
        self.conv_dim = 2 * self.kd + self.vd
        g = ex.get
        self.wqkv = ops.matrix(dev, g(p + "in_proj_qkv.weight"), self.qkv_fmt)
        self.wz = ops.matrix(dev, g(p + "in_proj_z.weight"), self.z_fmt)
        self.wab = ops.matrix(
            dev, torch.cat([g(p + "in_proj_a.weight"), g(p + "in_proj_b.weight")], 0), self.ab_fmt
        )
        self.wout = ops.matrix(dev, g(p + "out_proj.weight"), self.out_fmt)
        self.norm = ops.norm_weight(dev, g(p + "norm.weight"))

        def row(t, dtype=ttnn.bfloat16):
            return ttnn.as_tensor(
                t.reshape(1, 1, 1, -1).float(),
                dtype=dtype,
                layout=ttnn.TILE_LAYOUT,
                device=dev,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )

        w = g(p + "conv1d.weight").float().reshape(self.conv_dim, c.conv)  # [C, 1, K] depthwise
        self.taps = [row(w[:, j]) for j in range(c.conv)]
        self.a_neg = row(-g(p + "A_log").float().exp(), ttnn.float32)
        self.dt_bias = row(g(p + "dt_bias").float(), ttnn.float32)
        zeros = {"dtype": ttnn.bfloat16, "layout": ttnn.TILE_LAYOUT, "device": dev}
        # the conv's last conv - 1 inputs, oldest first, one logical row each (decode)
        self.conv_state = [
            ttnn.zeros([1, 1, 1, self.conv_dim], memory_config=ttnn.DRAM_MEMORY_CONFIG, **zeros)
            for _ in range(c.conv - 1)
        ]
        self.state = ttnn.zeros(
            [1, c.v_heads, c.k_dim, c.v_dim],
            dtype=ttnn.float32,
            layout=ttnn.TILE_LAYOUT,
            device=dev,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    def _gates(self, h, n):
        """g and beta [1, n, v_heads] fp32 from h's rows: a and b in one matmul."""
        import ttnn

        c = self.model.cfg
        ab = ops.linear(h, self.wab, ops.fidelity_for(self.ab_fmt))
        rows = n
        a = ttnn.slice(ab, [0, 0, 0, 0], [1, 1, rows, c.v_heads])
        b = ttnn.slice(ab, [0, 0, 0, c.v_heads], [1, 1, rows, 2 * c.v_heads])
        ttnn.deallocate(ab)
        a = ttnn.typecast(a, ttnn.float32)
        sp = ttnn.softplus(ttnn.add(a, self.dt_bias))
        gdec = ttnn.multiply(sp, self.a_neg)
        beta = ttnn.sigmoid(ttnn.typecast(b, ttnn.float32))
        for t in (a, b, sp):
            ttnn.deallocate(t)
        return ttnn.reshape(gdec, (1, rows, c.v_heads)), ttnn.reshape(beta, (1, rows, c.v_heads))

    def _out(self, o, z):
        """RMSNorm per value head (weight as stored), times silu(z), then out_proj. o and z are
        [1, rows, v_heads, v_dim] tile tensors."""
        import ttnn

        c = self.model.cfg
        y = ops.rms_norm(o, self.norm, c.eps)
        y = ttnn.multiply(z, y, input_tensor_a_activations=[ttnn.UnaryOpType.SILU])
        rows = y.shape[1]
        y = ttnn.reshape(y, (1, 1, rows, self.vd))
        out = ops.linear(y, self.wout, ops.fidelity_for(self.out_fmt))
        ttnn.deallocate(y)
        return out

    def prefill(self, h, cos, sin, n):
        """h [1, 1, T, dim] from position 0, of which the first n rows are real (T a multiple of
        32). Padded rows get beta 0 and g 0, identity steps of the state."""
        import ttnn

        c = self.model.cfg
        T = h.shape[-2]
        qkv = ops.linear(h, self.wqkv, ops.fidelity_for(self.qkv_fmt))
        z = ops.linear(h, self.wz, ops.fidelity_for(self.z_fmt))
        gdec, beta = self._gates(h, T)
        if n < T:  # padded positions: identity updates
            keep = torch.zeros(1, T, c.v_heads)
            keep[0, :n] = 1.0
            m = ttnn.from_torch(
                keep, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=h.device()
            )
            gdec, beta = ttnn.multiply(gdec, m), ttnn.multiply(beta, m)
        rm = ttnn.to_layout(qkv, ttnn.ROW_MAJOR_LAYOUT)
        ttnn.deallocate(qkv)
        # the conv's state for decode: the last conv - 1 real inputs (zeros before the prompt)
        for j in range(c.conv - 1):
            r = n - (c.conv - 1) + j
            if r >= 0:
                row = ttnn.slice(rm, [0, 0, r, 0], [1, 1, r + 1, self.conv_dim])
                ttnn.copy(ttnn.to_layout(row, ttnn.TILE_LAYOUT), self.conv_state[j])
            else:
                ttnn.copy(ttnn.zeros_like(self.conv_state[j]), self.conv_state[j])
        x = self._conv_prefill(rm)
        ttnn.deallocate(rm)
        # q and k to unit L2 norm per head, as decode and verify do
        q = self._l2(ttnn.slice(x, [0, 0, 0, 0], [1, 1, T, self.kd]), c.k_heads, T)
        k = self._l2(ttnn.slice(x, [0, 0, 0, self.kd], [1, 1, T, 2 * self.kd]), c.k_heads, T)
        v = ttnn.slice(x, [0, 0, 0, 2 * self.kd], [1, 1, T, self.conv_dim])
        ttnn.deallocate(x)
        o, st = ttnn.transformer.chunk_gated_delta_rule(
            ttnn.reshape(q, (1, T, self.kd)),
            ttnn.reshape(k, (1, T, self.kd)),
            ttnn.reshape(v, (1, T, self.vd)),
            gdec,
            beta,
            chunk_size=32,
            output_final_state=True,
        )
        ttnn.copy(st, self.state)
        ttnn.deallocate(st)
        o = ttnn.to_layout(ttnn.typecast(o, ttnn.bfloat16), ttnn.TILE_LAYOUT)  # [1, T, HV, V]
        z = ttnn.reshape(z, (1, T, c.v_heads, c.v_dim))
        return self._out(o, z)

    def _conv_prefill(self, rm):
        """The causal depthwise conv over rm [1, 1, T, C] (row major) from a zero history, then
        SiLU: the sum over taps j of tap_j * rm shifted down by (conv - 1 - j) rows."""
        import ttnn

        c = self.model.cfg
        T, C, k = rm.shape[-2], rm.shape[-1], c.conv
        pad = ttnn.zeros(
            [1, 1, k - 1, C], dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=rm.device()
        )
        xp = ttnn.concat([pad, rm], dim=2)
        out = None
        for j in range(k):
            s = ttnn.to_layout(ttnn.slice(xp, [0, 0, j, 0], [1, 1, j + T, C]), ttnn.TILE_LAYOUT)
            term = ttnn.multiply(s, self.taps[j])
            out = term if out is None else ttnn.add(out, term)
        return ttnn.silu(out)

    def decode(self, h, cos, sin, pos, mcs):
        """h [1, 1, 1 (32), dim]: one step of the conv and of the recurrent state. More rows (a
        verify's T draft rows) go to _verify, which keeps every row's state until commit()."""
        import ttnn

        if mcs["users"] > 1:
            return self._verify(h, mcs["users"])
        c = self.model.cfg
        qkv = ops.linear(h, self.wqkv, ops.fidelity_for(self.qkv_fmt))
        z = ops.linear(h, self.wz, ops.fidelity_for(self.z_fmt))
        gdec, beta = self._gates(h, 1)
        k = c.conv
        qkv = ops.rows(qkv, 1)
        acc = ttnn.multiply(qkv, self.taps[k - 1])
        for j in range(k - 1):
            acc = ttnn.add(acc, ttnn.multiply(self.conv_state[j], self.taps[j]))
        for j in range(k - 2):
            ttnn.copy(self.conv_state[j + 1], self.conv_state[j])
        ttnn.copy(qkv, self.conv_state[k - 2])
        ttnn.deallocate(qkv)
        x = ttnn.silu(acc)
        q = self._l2(ttnn.slice(x, [0, 0, 0, 0], [1, 1, 1, self.kd]), c.k_heads)
        kk = self._l2(ttnn.slice(x, [0, 0, 0, self.kd], [1, 1, 1, 2 * self.kd]), c.k_heads)
        v = ttnn.reshape(
            ttnn.slice(x, [0, 0, 0, 2 * self.kd], [1, 1, 1, self.conv_dim]),
            (1, 1, c.v_heads, c.v_dim),
        )
        o, st = ttnn.transformer.fused_recurrent_gated_delta_rule(
            q, kk, v, gdec, beta, initial_state=self.state, output_final_state=True
        )
        ttnn.copy(st, self.state)
        ttnn.deallocate(st)
        o = ttnn.to_layout(ttnn.typecast(o, ttnn.bfloat16), ttnn.TILE_LAYOUT)  # [1, 1, HV, V]
        z = ttnn.reshape(z, (1, 1, c.v_heads, c.v_dim))
        return self._out(o, z)

    def _verify(self, h, T):
        """T rows at consecutive positions after the state: the conv over the history and the
        rows, the Gated Delta Rule with the state after every row kept (output_per_token_state);
        self.state and the conv history stay until commit(a) keeps the first a rows."""
        import ttnn

        c = self.model.cfg
        C, k = self.conv_dim, c.conv
        qkv = ops.linear(h, self.wqkv, ops.fidelity_for(self.qkv_fmt))
        z = ops.linear(h, self.wz, ops.fidelity_for(self.z_fmt))
        gdec, beta = self._gates(h, T)
        rm = ttnn.to_layout(ops.rows(qkv, T), ttnn.ROW_MAJOR_LAYOUT)
        ttnn.deallocate(qkv)
        hist = [ttnn.to_layout(s, ttnn.ROW_MAJOR_LAYOUT) for s in self.conv_state]
        win = ttnn.concat([*hist, rm], dim=2)  # [1, 1, conv - 1 + T, C], oldest first
        out = None
        for j in range(k):
            s = ttnn.to_layout(ttnn.slice(win, [0, 0, j, 0], [1, 1, j + T, C]), ttnn.TILE_LAYOUT)
            term = ttnn.multiply(s, self.taps[j])
            out = term if out is None else ttnn.add(out, term)
        x = ttnn.silu(out)
        q = self._l2(ttnn.slice(x, [0, 0, 0, 0], [1, 1, T, self.kd]), c.k_heads, T)
        kk = self._l2(ttnn.slice(x, [0, 0, 0, self.kd], [1, 1, T, 2 * self.kd]), c.k_heads, T)
        v = ttnn.reshape(
            ttnn.slice(x, [0, 0, 0, 2 * self.kd], [1, 1, T, C]), (1, T, c.v_heads, c.v_dim)
        )
        o, states = ttnn.transformer.fused_recurrent_gated_delta_rule(
            q, kk, v, gdec, beta, initial_state=self.state, output_per_token_state=True
        )
        self._pending = (win, states)
        o = ttnn.to_layout(ttnn.typecast(o, ttnn.bfloat16), ttnn.TILE_LAYOUT)  # [1, T, HV, V]
        z = ttnn.reshape(z, (1, T, c.v_heads, c.v_dim))
        return self._out(o, z)

    def commit(self, a):
        """Keep the first a rows of the last verify: the state after row a - 1 and the conv's
        history up to it."""
        import ttnn

        c = self.model.cfg
        win, states = self._pending
        T = states.shape[1]
        flat = ttnn.reshape(states, (T, c.v_heads, c.k_dim, c.v_dim))
        ttnn.copy(ttnn.slice(flat, [a - 1, 0, 0, 0], [a, c.v_heads, c.k_dim, c.v_dim]), self.state)
        for j in range(c.conv - 1):  # win's rows a .. a + conv - 2 are the last conv - 1 inputs
            row = ttnn.slice(win, [0, 0, a + j, 0], [1, 1, a + j + 1, self.conv_dim])
            ttnn.copy(ttnn.to_layout(row, ttnn.TILE_LAYOUT), self.conv_state[j])
        ttnn.deallocate(states)
        ttnn.deallocate(win)
        self._pending = None

    def _l2(self, x, heads, rows=1):
        """x [1, 1, rows, heads * K] as [1, rows, heads, K], each head scaled to unit L2 norm
        (HF's l2norm, eps 1e-6): rms_norm with eps / K, divided by sqrt(K)."""
        import ttnn

        kdim = x.shape[-1] // heads
        y = ttnn.reshape(x, (1, rows, heads, kdim))
        y = ttnn.rms_norm(y, epsilon=1e-6 / kdim, compute_kernel_config=ops.compute_config("hifi4"))
        return ttnn.multiply(y, 1.0 / math.sqrt(kdim))


class Qwen35Layer:
    def __init__(self, model, i, prefix=None, fmt=None):
        """Decoder layer i, or with prefix the MTP's attention layer (its matrices at fmt)."""
        ex, dev = model.export, model.device
        p = prefix or f"{ex.prefix}layers.{i}."
        self.model, self.i = model, i
        self.kind = ATTN if prefix else model.cfg.types[i]
        if self.kind == GDN:
            self.mix = GatedDeltaNet(model, i)
        else:
            self.mix = GatedAttention(model, i, prefix, fmt)
        self.attn_norm = ops.norm_weight(dev, ex.get(p + "input_layernorm.weight"), 1.0)
        self.mlp_norm = ops.norm_weight(dev, ex.get(p + "post_attention_layernorm.weight"), 1.0)
        f = ex.formats
        self.mlp = ops.SwiGLU(
            dev,
            ex.get(p + "mlp.gate_proj.weight"),
            ex.get(p + "mlp.up_proj.weight"),
            ex.get(p + "mlp.down_proj.weight"),
            [fmt or f[(i, f"mlp.{n}_proj")] for n in ("gate", "up", "down")],
        )

    def _rest(self, x, y):
        import ttnn

        c = self.model.cfg
        n = x.shape[-2]  # both sides of each add at x's rows
        s = ttnn.add(x, ops.rows(y, n))
        ttnn.deallocate(y)
        ttnn.deallocate(x)  # the layer's input, made by the step before it
        h = ops.rms_norm(s, self.mlp_norm, c.eps)
        m = ops.rows(self.mlp(h), n)
        ttnn.deallocate(h)
        out = ttnn.add(s, m)
        ttnn.deallocate(m)
        ttnn.deallocate(s)
        return out

    def prefill(self, x, cos, sin, n=None):
        import ttnn

        h = ops.rms_norm(x, self.attn_norm, self.model.cfg.eps)
        y = self.mix.prefill(h, cos, sin, x.shape[-2] if n is None else n)
        ttnn.deallocate(h)
        return self._rest(x, y)

    def decode(self, x, cos, sin, pos, mcs):
        import ttnn

        h = ops.rows(ops.rms_norm(x, self.attn_norm, self.model.cfg.eps), mcs["users"])
        y = self.mix.decode(h, cos, sin, pos, mcs)
        ttnn.deallocate(h)
        return self._rest(x, y)


class Qwen35:
    """Qwen3.5 on one chip; the same interface as Qwen3 (prefill, decode, pcc's layer access)."""

    def __init__(self, export, device, max_seq=1024, layers=None, head=True, trace=False):
        if trace:
            raise NotImplementedError("Qwen3.5 traced decode cannot preserve recurrent state yet")

        import ttnn

        from .qwen3 import Qwen3

        if device.arch() != ttnn.device.Arch.BLACKHOLE:
            raise NotImplementedError(
                "Qwen3.5 needs ttnn's Gated Delta Rule ops, which run on Blackhole only for now"
            )
        self.export, self.device, self.max_seq = export, device, max_seq
        self.trace, self._trace = trace, None
        ops.set_arch(device)
        self.cfg = c = Config(export.text_config)
        assert max_seq % TILE == 0
        self.perm = head_perm(c.head_dim, c.rotary)
        self.rope = Qwen35Rope(device, c, max_seq)
        ids = range(c.layers) if layers is None else layers
        self.layers = {i: Qwen35Layer(self, i) for i in ids}
        self._mcs = {}
        self.mcs = Qwen3.decode_configs(self, 1)
        self.head = self.khead = None
        if head:
            ex = export
            self.embedding = ttnn.as_tensor(
                ex.get(ex.embed_name).to(torch.bfloat16),
                dtype=ttnn.bfloat16,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            self.final_norm = ops.norm_weight(device, ex.get(f"{ex.prefix}norm.weight"), 1.0)
            self.head, self.khead = head_for(self, ex, device)

    def prefill_layers(self, x, n=None):
        cos, sin = self.rope.prefill(0, x.shape[-2])
        for i in sorted(self.layers):
            x = self.layers[i].prefill(x, cos, sin, n)
        return x


class Mtp:
    """Qwen3.5's multi-token prediction module, drafting on the device: fc over
    [pre_fc_norm_embedding(embedding of the next token); pre_fc_norm_hidden(the model's row after
    its final norm)], one gated full-attention layer with its own paged KV cache, mtp.norm, then
    the model's head (not its final norm) and argmax, as vLLM's Qwen3-Next MTP. Its matrices are
    bf16 in the export (quant4.json does not list them) and are packed at fmt (bf16 by default:
    exact)."""

    def __init__(self, model, fmt="bf16"):
        ex, dev = model.export, model.device
        self.model = model
        self.fc = ops.matrix(dev, ex.get("mtp.fc.weight"), fmt)
        self.norm_e = ops.norm_weight(dev, ex.get("mtp.pre_fc_norm_embedding.weight"), 1.0)
        self.norm_h = ops.norm_weight(dev, ex.get("mtp.pre_fc_norm_hidden.weight"), 1.0)
        self.norm = ops.norm_weight(dev, ex.get("mtp.norm.weight"), 1.0)
        self.layer = Qwen35Layer(model, -1, prefix="mtp.layers.0.", fmt=fmt)

    def rows(self, hidden, tokens, pos, chained=False):
        """hidden [1, 1, n, dim]: the model's last-layer rows (before its final norm, which is
        applied here) at positions pos .. pos + n - 1, or with chained the MTP's own normed rows of
        the step before; tokens: the token after each. The rows go through fc, the MTP layer
        (writing its KV at those positions: a prompt from position 0 as a prefill, at most 8 rows
        later as a decode of n users) and mtp.norm. Returns (the normed rows, the draft id after
        the last row)."""
        import ttnn

        m, c = self.model, self.model.cfg
        n = len(tokens)
        prompt = pos == 0 and n > 1
        e = m.embed(m.ids(tokens), rows=None if prompt else n)
        # the hidden rows after the model's final norm, as vLLM and q8 feed the MTP (KL against
        # the model's next row 0.65 with it, 0.82 without, in transformers on a 69-token text); a
        # chained step's rows are the MTP's own, already normed
        hn = hidden if chained else ops.rms_norm(hidden, m.final_norm, c.eps)
        cat = ttnn.concat(
            [ops.rms_norm(e, self.norm_e, c.eps), ops.rms_norm(hn, self.norm_h, c.eps)], dim=-1
        )
        ttnn.deallocate(e)
        if not chained:
            ttnn.deallocate(hn)
        x = ops.linear(cat, self.fc)
        ttnn.deallocate(cat)
        if prompt:
            cos, sin = m.rope.prefill(0, x.shape[-2])
            y = self.layer.prefill(x, cos, sin, n)
        else:
            mcs = m.decode_configs(n)
            p = torch.arange(pos, pos + n, dtype=torch.int32)
            cur = ttnn.from_torch(p, dtype=ttnn.int32, device=m.device)
            idx = ttnn.from_torch(
                p.reshape(1, n), dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=m.device
            )
            cos, sin = m.rope.decode(idx, mcs["rope"])
            y = self.layer.decode(x, cos, sin, cur, mcs)
        y = ops.rms_norm(y, self.norm, c.eps)
        # the MTP's own norm, then the shared head (not the model's final norm)
        draft = m.greedy_normed(ttnn.slice(y, [0, 0, n - 1, 0], [1, 1, n, c.dim]), 1)[0]
        return y, draft

    def draft(self, hidden, tokens, pos, k):
        """rows(), then k - 1 chained steps, each on the previous call's last normed row and its
        draft: the k draft tokens after tokens[-1]."""
        import ttnn

        y, d = self.rows(hidden, tokens, pos)
        out, p, n = [d], pos + len(tokens), len(tokens)
        for _ in range(k - 1):
            # a slice of the whole tensor may be the tensor itself: y is freed after its use
            last = ttnn.slice(y, [0, 0, n - 1, 0], [1, 1, n, y.shape[-1]])
            y_next, d = self.rows(last, [d], p, chained=True)
            ttnn.deallocate(y)
            y = y_next
            out.append(d)
            p, n = p + 1, 1
        ttnn.deallocate(y)
        return out


def _borrow():
    """Qwen35 takes Qwen3's device-side helpers (ids, embed, positions, decode, head, prefill,
    traces), which only touch the attributes both classes set."""
    from .qwen3 import Qwen3

    for name in (
        "ids",
        "embed",
        "position",
        "decode_layers",
        "_decode_layers",
        "_top1",
        "_head_top1",
        "greedy",
        "greedy_normed",
        "_decode_step",
        "decode",
        "decode_traced",
        "release_trace",
        "decode_configs",
        "verify",
        "kv_source",
        "check_prefill",
        "check_decode",
    ):
        setattr(Qwen35, name, getattr(Qwen3, name))

    def prefill_hidden(self, tokens):
        """The prompt's KV rows and states, the greedy token after it, and the last layer's rows
        [1, 1, T, dim] (before the final norm; the caller frees them)."""
        import ttnn

        x = self.prefill_layers(self.embed(self.ids(tokens)), len(tokens))
        last = len(tokens) - 1
        row = ttnn.slice(x, [0, 0, last, 0], [1, 1, last + 1, self.cfg.dim])
        return self.greedy(row, 1)[0], x

    def prefill(self, tokens):
        import ttnn

        out, x = self.prefill_hidden(tokens)
        ttnn.deallocate(x)
        return out

    Qwen35.prefill_hidden = prefill_hidden
    Qwen35.prefill = prefill

    def commit(self, a):
        """Keep the first a rows of the last verify (the attention layers' rows past them are
        written over later)."""
        for layer in self.layers.values():
            if layer.kind == GDN:
                layer.mix.commit(a)

    Qwen35.commit = commit


_borrow()
