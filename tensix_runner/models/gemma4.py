# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Xylolabs
"""Gemma 4's text models with per-layer inputs (gemma-4-E2B, gemma-4-E4B) on one chip.

Layers alternate sliding-window attention (head_dim, RoPE over the whole head) and full attention
(global_head_dim, proportional RoPE: only the first frequencies are nonzero, so rotate_half over
the whole head with cos 1 and sin 0 elsewhere is exact). q, k and v each get an RMSNorm per head
(v's without a weight) and the attention scale is 1. The last num_kv_shared_layers layers have
no k or v projection: each attends over the KV cache of the last earlier layer of its type, which
the KV-sharing layer reads after that layer has written the step's row.

Each layer: input norm, attention, post-attention norm, residual add; pre-feedforward norm, a GELU
(tanh) gated MLP, post-feedforward norm, residual add; then gelu(per_layer_input_gate(x)) times the
layer's per-layer input, per_layer_projection, a norm, residual add; the sum scaled by
layer_scalar.

A layer's per-layer input: its slice of embed_tokens_per_layer at the token (times
sqrt(hidden_size_per_layer_input)) plus the RMSNorm of its slice of per_layer_model_projection
applied to the scaled embedding (times 1 / sqrt(hidden_size)), the sum times 1 / sqrt(2). Each
layer's slices are device tensors of their own, looked up and multiplied on the device.

The embedding is scaled by sqrt(hidden_size). The head's logit softcap (tanh(x / cap) * cap)
keeps the logits' order, so the greedy argmax runs without it."""

import math

import torch

from .. import ops
from ..kv import PagedKV
from ..rope import HfRope, inv_freq
from .qwen3 import TILE, Qwen3, head_for

SLIDING, FULL = "sliding_attention", "full_attention"


class Config:
    def __init__(self, tc):
        self.dim = tc["hidden_size"]
        self.layers = tc["num_hidden_layers"]
        self.types = tc["layer_types"]
        self.heads = tc["num_attention_heads"]
        self.kv_heads = tc["num_key_value_heads"]
        self.head_dim = tc["head_dim"]
        self.global_head_dim = tc.get("global_head_dim") or self.head_dim
        self.eps = tc["rms_norm_eps"]
        self.vocab = tc["vocab_size"]
        self.window = tc["sliding_window"]
        self.first_shared = self.layers - (tc.get("num_kv_shared_layers") or 0)
        self.ple_dim = tc.get("hidden_size_per_layer_input") or 0
        self.softcap = tc.get("final_logit_softcapping")
        self.rope = tc["rope_parameters"]
        for key, ok in (
            ("attention_k_eq_v", False),
            ("enable_moe_block", False),
            ("attention_bias", False),
            ("hidden_activation", "gelu_pytorch_tanh"),
        ):
            if tc.get(key, ok) != ok:
                raise NotImplementedError(f"Gemma 4 with {key} = {tc[key]!r}")
        if tc.get("use_bidirectional_attention"):
            raise NotImplementedError("Gemma 4 with bidirectional attention")
        if not self.ple_dim:
            raise NotImplementedError("Gemma 4 without per-layer inputs (only the E models)")

    def head_dim_of(self, kind):
        return self.global_head_dim if kind == FULL else self.head_dim

    def source(self, i):
        """The layer whose KV cache layer i attends over: i itself, or for a KV-sharing layer the
        last layer of its type before the first KV-sharing one (as transformers picks it)."""
        if i < self.first_shared:
            return i
        kind = self.types[i]
        return max(j for j in range(self.first_shared) if self.types[j] == kind)

    def inv_freq(self, kind):
        """The inverse frequencies [head_dim / 2] of a layer type's RoPE, as transformers computes
        them ("default", or "proportional": the first partial_rotary_factor of them, then zeros)."""
        p, d = self.rope[kind], self.head_dim_of(kind)
        theta, how = p["rope_theta"], p.get("rope_type", "default")
        if how == "default":
            return inv_freq(d, theta)
        if how == "proportional":
            n = int(p.get("partial_rotary_factor", 1.0) * d // 2)
            f = 1.0 / (theta ** (torch.arange(0, 2 * n, 2, dtype=torch.int64).float() / d))
            return torch.cat([f, torch.zeros(d // 2 - n)]) / p.get("factor", 1.0)
        raise NotImplementedError(f"RoPE type {how!r}")


class Gemma4Layer:
    def __init__(self, model, i):
        ex, dev, c = model.export, model.device, model.cfg
        self.model, self.i = model, i
        self.type, self.hd, self.src = c.types[i], c.head_dim_of(c.types[i]), c.source(i)
        p = f"{ex.prefix}layers.{i}."
        g = ex.get

        def fmt(proj):
            return ex.formats.get((i, proj), "bf16")

        if self.src == i:
            fmts = {fmt(f"self_attn.{n}_proj") for n in "qkv"}
            assert len(fmts) == 1, f"layer {i}: q, k and v in different formats {fmts}"
            self.q_fmt = fmts.pop()
            w = torch.cat([g(p + f"self_attn.{n}_proj.weight") for n in "qkv"], 0)
            self.k_norm = ops.norm_weight(dev, g(p + "self_attn.k_norm.weight"))
            self.kv = PagedKV(dev, c.kv_heads, self.hd, model.max_seq)
        else:  # KV-sharing: q only
            self.q_fmt = fmt("self_attn.q_proj")
            w = g(p + "self_attn.q_proj.weight")
            self.kv = None
        self.wq = ops.matrix(dev, w, self.q_fmt)
        self.q_norm = ops.norm_weight(dev, g(p + "self_attn.q_norm.weight"))
        self.o_fmt = fmt("self_attn.o_proj")
        self.wo = ops.matrix(dev, g(p + "self_attn.o_proj.weight"), self.o_fmt)
        self.norms = {
            n: ops.norm_weight(dev, g(p + f"{n}.weight"))
            for n in (
                "input_layernorm",
                "post_attention_layernorm",
                "pre_feedforward_layernorm",
                "post_feedforward_layernorm",
                "post_per_layer_input_norm",
            )
        }
        self.mlp = ops.GeluTanhMLP(
            dev,
            g(p + "mlp.gate_proj.weight"),
            g(p + "mlp.up_proj.weight"),
            g(p + "mlp.down_proj.weight"),
            [fmt(f"mlp.{n}_proj") for n in ("gate", "up", "down")],
        )
        self.ple_fmts = [fmt(n) for n in ("per_layer_input_gate", "per_layer_projection")]
        self.ple_gate = ops.matrix(dev, g(p + "per_layer_input_gate.weight"), self.ple_fmts[0])
        self.ple_proj = ops.matrix(dev, g(p + "per_layer_projection.weight"), self.ple_fmts[1])
        self.scalar = float(g(p + "layer_scalar").float().reshape(-1)[0])

    @property
    def window(self):
        return self.model.cfg.window if self.type == SLIDING else None

    def _norm(self, x, name):
        return ops.rms_norm(x, self.norms[name], self.model.cfg.eps)

    def _head_norm(self, x, w, decode):
        """An RMSNorm over each head's head_dim (w None: without a weight); decode heads are height
        sharded."""
        import ttnn

        eps = self.model.cfg.eps
        if not decode:
            return ops.rms_norm(x, w, eps)
        mc = x.memory_config()
        y = ops.rms_norm(
            ttnn.to_memory_config(x, ttnn.L1_MEMORY_CONFIG), w, eps, ttnn.L1_MEMORY_CONFIG
        )
        return ttnn.to_memory_config(y, mc)

    def prefill(self, x, cos, sin, ple):
        """x [1, 1, T, dim] from position 0 and the layer's per-layer input [1, 1, T, ple_dim];
        fills the KV cache's first T rows (a layer with its own k and v)."""
        import ttnn

        c, m = self.model.cfg, self.model
        h = self._norm(x, "input_layernorm")
        qkv = ops.linear(h, self.wq, ops.fidelity_for(self.q_fmt))
        ttnn.deallocate(h)
        if self.kv is not None:
            q, k, v = ttnn.experimental.nlp_create_qkv_heads(
                qkv,
                num_heads=c.heads,
                num_kv_heads=c.kv_heads,
                transpose_k_heads=False,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            k = self._head_norm(k, self.k_norm, False)
            v = self._head_norm(v, None, False)
            k = ttnn.experimental.rotary_embedding_hf(k, cos, sin, is_decode_mode=False)
            self.kv.fill(k, v)
            if self.i in m.kv_sources:  # a later layer attends over these
                for t in m.prefill_kv.pop(self.i, ()):
                    ttnn.deallocate(t)
                m.prefill_kv[self.i] = (k, v)
        else:
            T = qkv.shape[-2]
            q = ttnn.transpose(ttnn.reshape(qkv, (1, T, c.heads, self.hd)), 1, 2)
            k, v = m.prefill_kv[self.src]
        ttnn.deallocate(qkv)
        q = self._head_norm(q, self.q_norm, False)
        q = ttnn.experimental.rotary_embedding_hf(q, cos, sin, is_decode_mode=False)
        a = ttnn.transformer.scaled_dot_product_attention(
            q,
            k,
            v,
            is_causal=True,
            scale=1.0,
            sliding_window_size=self.window,
            compute_kernel_config=ops.compute_config("hifi4"),
            program_config=ops.sdpa_config(self.model.device),
        )
        ttnn.deallocate(q)
        if self.kv is not None and self.i not in m.kv_sources:
            ttnn.deallocate(k)
            ttnn.deallocate(v)
        o = ttnn.experimental.nlp_concat_heads(a, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(a)
        return self._rest(x, o, ple)

    def decode(self, x, cos, sin, pos, mcs, ple):
        """x [1, 1, users, dim] at the positions in pos (int32 [users] on the device), with the
        layer's per-layer input [1, 1, users, ple_dim]."""
        import ttnn

        c = self.model.cfg
        users, sh = mcs["users"], mcs[self.hd]
        h = ops.rows(self._norm(x, "input_layernorm"), users)
        qkv = ops.linear(h, self.wq, ops.fidelity_for(self.q_fmt), ttnn.L1_MEMORY_CONFIG)
        ttnn.deallocate(h)
        w = qkv.shape[-1]
        qkv = ttnn.reshape(qkv, (1, 1, users, w), (1, 1, TILE, w))
        if self.kv is not None:
            q, k, v = ttnn.experimental.nlp_create_qkv_heads_decode(
                qkv, num_heads=c.heads, num_kv_heads=c.kv_heads, memory_config=sh["heads"]
            )
            ttnn.deallocate(qkv)
            k = self._head_norm(k, self.k_norm, True)
            v = self._head_norm(v, None, True)
            k = ttnn.experimental.rotary_embedding_hf(k, cos, sin, is_decode_mode=True)
            self.kv.update(k, v, pos, users)
            ttnn.deallocate(k)
            ttnn.deallocate(v)
            kv = self.kv
        else:  # q only: [1, users, heads, head_dim], sharded as nlp_create_qkv_heads_decode does
            q = ttnn.interleaved_to_sharded(
                ttnn.reshape(qkv, (1, users, c.heads, self.hd)), sh["heads"]
            )
            ttnn.deallocate(qkv)
            kv = self.model.layers[self.src].kv
        q = self._head_norm(q, self.q_norm, True)
        q = ttnn.experimental.rotary_embedding_hf(q, cos, sin, is_decode_mode=True)
        a = kv.attend(q, pos, 1.0, ops.compute_config("hifi4"), users, self.window)
        ttnn.deallocate(q)
        a = ttnn.to_memory_config(a, sh["sdpa_out"])
        o = ttnn.experimental.nlp_concat_heads_decode(a, num_heads=c.heads)
        ttnn.deallocate(a)
        o = ops.rows(ttnn.to_memory_config(o, ttnn.DRAM_MEMORY_CONFIG), users)
        return self._rest(x, o, ple)

    def _rest(self, x, o, ple):
        """The output projection, the MLP and the per-layer input, each normed and added."""
        import ttnn

        n = x.shape[-2]  # both sides of each add at x's rows

        def add(a, b):
            out = ttnn.add(a, b)
            ttnn.deallocate(a)
            ttnn.deallocate(b)
            return out

        y = ops.rows(ops.linear(o, self.wo, ops.fidelity_for(self.o_fmt)), n)
        ttnn.deallocate(o)
        s = add(x, self._norm_free(y, "post_attention_layernorm"))
        h = self._norm(s, "pre_feedforward_layernorm")
        m = ops.rows(self.mlp(h), n)
        ttnn.deallocate(h)
        s = add(s, self._norm_free(m, "post_feedforward_layernorm"))
        gate = ops.rows(ops.linear(s, self.ple_gate, ops.fidelity_for(self.ple_fmts[0])), n)
        a = ops.gelu_tanh(gate)
        ttnn.deallocate(gate)
        a2 = ttnn.multiply(a, ple)
        ttnn.deallocate(a)
        ttnn.deallocate(ple)
        y = ops.rows(ops.linear(a2, self.ple_proj, ops.fidelity_for(self.ple_fmts[1])), n)
        ttnn.deallocate(a2)
        s = add(s, self._norm_free(y, "post_per_layer_input_norm"))
        if self.scalar == 1.0:
            return s
        out = ttnn.multiply(s, self.scalar)
        ttnn.deallocate(s)
        return out

    def _norm_free(self, y, name):
        import ttnn

        out = self._norm(y, name)
        ttnn.deallocate(y)
        return out


class Gemma4(Qwen3):
    """Gemma 4 on one chip; the interface of Qwen3. layers: the decoder layers to load (default
    all), with the layers whose KV caches they attend over; the embedding and the per-layer input
    tables of those layers always load (the per-layer inputs read them), the final norm and the
    head with head=True."""

    def __init__(self, export, device, max_seq=1024, layers=None, head=True, trace=False):
        import ttnn

        self.export, self.device, self.max_seq = export, device, max_seq
        self.trace, self._trace = trace, None
        self.cfg = c = Config(export.text_config)
        assert max_seq % TILE == 0
        want = range(c.layers) if layers is None else layers
        ids = sorted(set(want) | {c.source(i) for i in want})
        kinds = {c.types[i] for i in ids}
        self.ropes = {
            k: HfRope(device, c.head_dim_of(k), max_seq, None, c.inv_freq(k)) for k in kinds
        }
        self.kv_sources = {c.source(i) for i in ids if c.source(i) != i}
        self.prefill_kv, self._ple = {}, {}
        self.layers = {i: Gemma4Layer(self, i) for i in ids}
        self._mcs = {}
        self.mcs = self.decode_configs(1)
        ex, p, d = export, export.prefix, c.ple_dim

        def table(w):
            return ttnn.as_tensor(
                w.to(torch.bfloat16).contiguous(),
                dtype=ttnn.bfloat16,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )

        self.embedding = table(ex.get(ex.embed_name))
        tab = ex.part(f"{p}embed_tokens_per_layer.weight")
        proj = ex.part(f"{p}per_layer_model_projection.weight")
        self.ple_tables = {i: table(tab[:, i * d : (i + 1) * d]) for i in ids}
        self.ple_w = {i: ops.matrix(device, proj[i * d : (i + 1) * d], "bf16") for i in ids}
        self.ple_norm = ops.norm_weight(device, ex.get(f"{p}per_layer_projection_norm.weight"))
        self.head = self.khead = None
        if head:
            self.final_norm = ops.norm_weight(device, ex.get(f"{p}norm.weight"))
            self.head, self.khead = head_for(self, ex, device)

    def kv_source(self, i):
        return self.cfg.source(i)

    def decode_configs(self, users=1):
        """The decode heads' height-sharded layouts for `users` rows, per head_dim: one core per
        user, as in Qwen3. Made once per user count."""
        import ttnn

        if users in self._mcs:
            return self._mcs[users]
        c = self.cfg
        grid = ops.user_grid(self.device, users)

        def shard(rows, hd):
            return ttnn.create_sharded_memory_config(
                shape=(rows, hd),
                core_grid=grid,
                strategy=ttnn.ShardStrategy.HEIGHT,
                orientation=ttnn.ShardOrientation.ROW_MAJOR,
                use_height_and_width_as_shard_shape=True,
            )

        out = {"users": users}
        for hd in {c.head_dim, c.global_head_dim}:
            out[hd] = {
                "heads": shard(TILE, hd),
                "sdpa_out": shard(math.ceil(c.heads / TILE) * TILE, hd),
            }
        self._mcs[users] = out
        return out

    # the per-layer inputs -----------------------------------------------------------------------

    def per_layer_inputs(self, ids, x0, rows=None, layers=None):
        """{layer: per-layer input [1, 1, n, ple_dim]} from the token ids and the scaled embedding
        x0 [1, 1, n, dim], for the loaded layers (or those in layers)."""
        import ttnn

        c = self.cfg
        n = x0.shape[-2]
        out = {}
        for i in sorted(self.layers) if layers is None else layers:
            t = ttnn.embedding(ids, self.ple_tables[i], layout=ttnn.TILE_LAYOUT)
            m = t.shape[-2]
            t = ttnn.reshape(t, (1, 1, n, c.ple_dim), (1, 1, m, c.ple_dim))
            tok = ttnn.multiply(t, float(c.ple_dim) ** 0.5)
            ttnn.deallocate(t)
            pr = ops.rows(ops.linear(x0, self.ple_w[i], "hifi4"), n)
            pr2 = ttnn.multiply(pr, float(c.dim) ** -0.5)
            ttnn.deallocate(pr)
            pr = ops.rms_norm(pr2, self.ple_norm, c.eps)
            ttnn.deallocate(pr2)
            s = ttnn.add(pr, tok)
            ttnn.deallocate(pr)
            ttnn.deallocate(tok)
            out[i] = ttnn.multiply(s, 2.0**-0.5)
            ttnn.deallocate(s)
        return out

    def embed(self, ids, rows=None):
        """The scaled embedding [1, 1, n, dim]; the loaded layers' per-layer inputs are made from
        the same ids and kept for the step's layers."""
        import ttnn

        x = Qwen3.embed(self, ids, rows)
        x0 = ttnn.multiply(x, float(self.cfg.dim) ** 0.5)
        ttnn.deallocate(x)
        self._ple = self.per_layer_inputs(ids, x0, rows)
        return x0

    def _inputs(self, tokens, rows, layers):
        import ttnn

        ids = self.ids(tokens)
        x = Qwen3.embed(self, ids, rows)
        x0 = ttnn.multiply(x, float(self.cfg.dim) ** 0.5)
        ttnn.deallocate(x)
        out = self.per_layer_inputs(ids, x0, rows, layers)
        ttnn.deallocate(x0)
        return out

    # the layers ---------------------------------------------------------------------------------

    def prefill_layers(self, x, n=None):
        import ttnn

        T = x.shape[-2]
        rope = {k: r.prefill(0, T) for k, r in self.ropes.items()}
        for i in sorted(self.layers):
            layer = self.layers[i]
            x = layer.prefill(x, *rope[layer.type], self._ple.pop(i))
        for kv in self.prefill_kv.values():
            for t in kv:
                ttnn.deallocate(t)
        self.prefill_kv.clear()
        return x

    def _decode_layers(self, x, cur, idx, users=1):
        mcs = self.decode_configs(users)
        rope = {k: r.decode(idx, mcs[r.head_dim]["heads"]) for k, r in self.ropes.items()}
        for i in sorted(self.layers):
            layer = self.layers[i]
            x = layer.decode(x, *rope[layer.type], cur, mcs, self._ple.pop(i))
        return x

    # pcc's layer access -------------------------------------------------------------------------

    def check_prefill(self, i, x, tokens):
        layer = self.layers[i]
        ple = self._inputs(tokens, None, [i])[i]
        cos, sin = self.ropes[layer.type].prefill(0, x.shape[-2])
        return layer.prefill(x, cos, sin, ple)

    def check_decode(self, i, x, token, p):
        layer = self.layers[i]
        ple = self._inputs([token], 1, [i])[i]
        cur, idx = self.position(p)
        rope = self.ropes[layer.type]
        cos, sin = rope.decode(idx, self.mcs[rope.head_dim]["heads"])
        return layer.decode(x, cos, sin, cur, self.mcs, ple)

    def per_layer_input(self, i, tokens):
        """Layer i's per-layer input rows [n, ple_dim] for tokens, read back (for pcc)."""
        import ttnn

        t = self._inputs(tokens, None, [i])[i]
        out = ttnn.to_torch(t).float().reshape(-1, self.cfg.ple_dim)[: len(tokens)]
        ttnn.deallocate(t)
        return out
