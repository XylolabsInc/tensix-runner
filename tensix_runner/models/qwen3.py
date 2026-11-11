# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Xylolabs
"""Qwen3 dense (Qwen3-4B, Qwen3-8B) on one chip.

Each decoder layer: RMSNorm, attention with grouped KV heads, a per-head RMSNorm on q and k
(q_norm, k_norm) and HF-style RoPE, then RMSNorm and a SwiGLU MLP, each with its residual add.
q, k and v are one matrix ([q | k | v] along the output, whole tiles: exact). The KV cache is one
bf16 tensor per layer, [1, kv_heads, max_seq, head_dim].

Prefill takes the prompt from position 0 in one pass (its length padded to a multiple of 32);
decode takes one token at a position. Both run eagerly here; the embedding lookup, the RoPE rows,
the head and its argmax run on the device, and only token ids and positions cross to it."""

import math

import torch

from .. import ops
from ..rope import HfRope

TILE = 32


class Qwen3Layer:
    def __init__(self, model, i):
        import ttnn

        ex, dev, c = model.export, model.device, model.cfg
        self.model, self.i = model, i
        p = f"{ex.prefix}layers.{i}."

        def fmt(proj):
            return ex.formats[(i, proj)]

        qkv_fmts = {fmt(f"self_attn.{n}_proj") for n in "qkv"}
        assert len(qkv_fmts) == 1, f"layer {i}: q, k and v in different formats {qkv_fmts}"
        self.qkv_fmt = qkv_fmts.pop()
        self.o_fmt = fmt("self_attn.o_proj")
        g = ex.get
        wqkv = torch.cat([g(p + f"self_attn.{n}_proj.weight") for n in "qkv"], 0)
        self.wqkv = ops.matrix(dev, wqkv, self.qkv_fmt)
        self.wo = ops.matrix(dev, g(p + "self_attn.o_proj.weight"), self.o_fmt)
        self.attn_norm = ops.norm_weight(dev, g(p + "input_layernorm.weight"))
        self.mlp_norm = ops.norm_weight(dev, g(p + "post_attention_layernorm.weight"))
        self.q_norm = ops.norm_weight(dev, g(p + "self_attn.q_norm.weight"))
        self.k_norm = ops.norm_weight(dev, g(p + "self_attn.k_norm.weight"))
        self.mlp = ops.SwiGLU(
            dev,
            g(p + "mlp.gate_proj.weight"),
            g(p + "mlp.up_proj.weight"),
            g(p + "mlp.down_proj.weight"),
            [fmt(f"mlp.{n}_proj") for n in ("gate", "up", "down")],
        )
        shape = [1, c.kv_heads, model.max_seq, c.head_dim]
        kw = {"dtype": ttnn.bfloat16, "layout": ttnn.TILE_LAYOUT, "device": dev}
        self.k_cache = ttnn.zeros(shape, memory_config=ttnn.DRAM_MEMORY_CONFIG, **kw)
        self.v_cache = ttnn.zeros(shape, memory_config=ttnn.DRAM_MEMORY_CONFIG, **kw)

    def _head_norm(self, x, w, decode):
        """q_norm or k_norm over each head's head_dim (decode heads are height sharded)."""
        import ttnn

        eps = self.model.cfg.eps
        if not decode:
            return ops.rms_norm(x, w, eps)
        mc = x.memory_config()
        y = ops.rms_norm(
            ttnn.to_memory_config(x, ttnn.L1_MEMORY_CONFIG), w, eps, ttnn.L1_MEMORY_CONFIG
        )
        return ttnn.to_memory_config(y, mc)

    def prefill(self, x, cos, sin):
        """x [1, 1, T, dim] from position 0; fills the KV cache's first T rows."""
        import ttnn

        c = self.model.cfg
        h = ops.rms_norm(x, self.attn_norm, c.eps)
        qkv = ops.linear(h, self.wqkv, ops.fidelity_for(self.qkv_fmt))
        ttnn.deallocate(h)
        q, k, v = ttnn.experimental.nlp_create_qkv_heads(
            qkv,
            num_heads=c.heads,
            num_kv_heads=c.kv_heads,
            transpose_k_heads=False,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(qkv)
        q = self._head_norm(q, self.q_norm, False)
        k = self._head_norm(k, self.k_norm, False)
        q = ttnn.experimental.rotary_embedding_hf(q, cos, sin, is_decode_mode=False)
        k = ttnn.experimental.rotary_embedding_hf(k, cos, sin, is_decode_mode=False)
        ttnn.fill_cache(self.k_cache, k, 0)
        ttnn.fill_cache(self.v_cache, v, 0)
        a = ttnn.transformer.scaled_dot_product_attention(
            q,
            k,
            v,
            is_causal=True,
            scale=1.0 / math.sqrt(c.head_dim),
            compute_kernel_config=ops.compute_config("hifi4"),
        )
        for t in (q, k, v):
            ttnn.deallocate(t)
        o = ttnn.experimental.nlp_concat_heads(a, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(a)
        return self._rest(x, o)

    def decode(self, x, cos, sin, pos, mcs):
        """x [1, 1, 32, dim] (row 0 the token) at the position in pos (int32 [1] on the device)."""
        import ttnn

        c = self.model.cfg
        h = ops.rms_norm(x, self.attn_norm, c.eps)
        qkv = ops.linear(h, self.wqkv, ops.fidelity_for(self.qkv_fmt), ttnn.L1_MEMORY_CONFIG)
        ttnn.deallocate(h)
        qkv = ttnn.reshape(qkv, (1, 1, 1, qkv.shape[-1]), (1, 1, TILE, qkv.shape[-1]))
        q, k, v = ttnn.experimental.nlp_create_qkv_heads_decode(
            qkv, num_heads=c.heads, num_kv_heads=c.kv_heads, memory_config=mcs["heads"]
        )
        ttnn.deallocate(qkv)
        q = self._head_norm(q, self.q_norm, True)
        k = self._head_norm(k, self.k_norm, True)
        q = ttnn.experimental.rotary_embedding_hf(q, cos, sin, is_decode_mode=True)
        k = ttnn.experimental.rotary_embedding_hf(k, cos, sin, is_decode_mode=True)
        ttnn.experimental.paged_update_cache(self.k_cache, k, update_idxs_tensor=pos)
        ttnn.experimental.paged_update_cache(self.v_cache, v, update_idxs_tensor=pos)
        ttnn.deallocate(k)
        ttnn.deallocate(v)
        a = ttnn.transformer.scaled_dot_product_attention_decode(
            q,
            self.k_cache,
            self.v_cache,
            cur_pos_tensor=pos,
            scale=1.0 / math.sqrt(c.head_dim),
            compute_kernel_config=ops.compute_config("hifi4"),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(q)
        a = ttnn.to_memory_config(a, mcs["sdpa_out"])
        o = ttnn.experimental.nlp_concat_heads_decode(a, num_heads=c.heads)
        ttnn.deallocate(a)
        o = ttnn.to_memory_config(o, ttnn.DRAM_MEMORY_CONFIG)
        return self._rest(x, o)

    def _rest(self, x, o):
        """The output projection, the residual adds and the MLP."""
        import ttnn

        c = self.model.cfg
        y = ops.linear(o, self.wo, ops.fidelity_for(self.o_fmt))
        ttnn.deallocate(o)
        x = ttnn.add(x, y)
        ttnn.deallocate(y)
        h = ops.rms_norm(x, self.mlp_norm, c.eps)
        m = self.mlp(h)
        ttnn.deallocate(h)
        out = ttnn.add(x, m)
        ttnn.deallocate(m)
        ttnn.deallocate(x)
        return out


class Config:
    def __init__(self, tc):
        self.dim = tc["hidden_size"]
        self.layers = tc["num_hidden_layers"]
        self.heads = tc["num_attention_heads"]
        self.kv_heads = tc["num_key_value_heads"]
        self.head_dim = tc.get("head_dim") or self.dim // self.heads
        self.eps = tc["rms_norm_eps"]
        self.theta = tc.get("rope_theta") or tc.get("rope_parameters", {}).get("rope_theta", 1e6)
        self.vocab = tc["vocab_size"]


class Qwen3:
    """The model on one chip. layers: the decoder layers to load (default all); the embedding, the
    final norm and the head load with head=True."""

    def __init__(self, export, device, max_seq=1024, layers=None, head=True):
        import ttnn

        self.export, self.device, self.max_seq = export, device, max_seq
        ops.set_arch(device)
        self.cfg = c = Config(export.text_config)
        assert max_seq % TILE == 0
        self.rope = HfRope(device, c.head_dim, max_seq, c.theta)
        ids = range(c.layers) if layers is None else layers
        self.layers = {i: Qwen3Layer(self, i) for i in ids}
        self.mcs = self._decode_configs()
        self.head = None
        if head:
            ex = export
            self.embedding = ttnn.as_tensor(
                ex.get(ex.embed_name).to(torch.bfloat16),
                dtype=ttnn.bfloat16,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            self.final_norm = ops.norm_weight(device, ex.get(f"{ex.prefix}norm.weight"))
            self.head = ops.matrix(device, ex.get(ex.head_name), ex.head_format)

    def _decode_configs(self):
        """The decode heads' height-sharded layouts for one user (as tt-transformers sets them)."""
        import ttnn

        c = self.cfg
        heads = ttnn.create_sharded_memory_config(
            shape=(TILE, c.head_dim),
            core_grid=ttnn.CoreRangeSet(
                {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))}
            ),
            strategy=ttnn.ShardStrategy.HEIGHT,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )
        sdpa_out = ttnn.create_sharded_memory_config(
            shape=(math.ceil(c.heads / TILE) * TILE, c.head_dim),
            core_grid=ttnn.CoreRangeSet(
                {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))}
            ),
            strategy=ttnn.ShardStrategy.HEIGHT,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )
        return {"heads": heads, "sdpa_out": sdpa_out, "rope": heads}

    # the device side of one token or one prompt -------------------------------------------------

    def ids(self, tokens):
        """Token ids as uint32 [1, n] on the device (n padded to a multiple of 32 with token 0)."""
        import ttnn

        n = -(-len(tokens) // TILE) * TILE
        t = torch.zeros(1, n, dtype=torch.int32)
        t[0, : len(tokens)] = torch.tensor(tokens, dtype=torch.int32)
        return ttnn.from_torch(
            t, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=self.device
        )

    def embed(self, ids, rows=None):
        """Rows [1, 1, n, dim] gathered on the device; with rows=1 (decode) one logical row in its
        tile of 32."""
        import ttnn

        x = ttnn.embedding(ids, self.embedding, layout=ttnn.TILE_LAYOUT)
        n, d = x.shape[-2], x.shape[-1]
        if rows is None:
            return ttnn.reshape(x, (1, 1, n, d))
        return ttnn.reshape(x, (1, 1, rows, d), (1, 1, n, d))

    def position(self, pos):
        """The decode position as int32 [1] (the KV cache's row) and uint32 [1, 1] (the RoPE
        gather's index), on the device."""
        import ttnn

        p = torch.tensor([pos], dtype=torch.int32)
        cur = ttnn.from_torch(p, dtype=ttnn.int32, device=self.device)
        idx = ttnn.from_torch(
            p.reshape(1, 1), dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=self.device
        )
        return cur, idx

    def prefill_layers(self, x):
        n = x.shape[-2]
        cos, sin = self.rope.prefill(0, n)
        for i in sorted(self.layers):
            x = self.layers[i].prefill(x, cos, sin)
        return x

    def decode_layers(self, x, pos):
        cur, idx = self.position(pos)
        cos, sin = self.rope.decode(idx, self.mcs["rope"])
        for i in sorted(self.layers):
            x = self.layers[i].decode(x, cos, sin, cur, self.mcs)
        return x

    def top1(self, x):
        """The greedy token of each of x's 32 rows [1, 1, 32, dim]: the final norm, the head and
        argmax on the device; the 32 ids are read back."""
        import ttnn

        h = ops.rms_norm(x, self.final_norm, self.cfg.eps)
        logits = ops.linear(h, self.head, ops.fidelity_for(self.export.head_format))
        ttnn.deallocate(h)
        logits = ttnn.to_layout(logits, ttnn.ROW_MAJOR_LAYOUT)
        top = ttnn.argmax(logits, dim=3)
        ttnn.deallocate(logits)
        return ttnn.to_torch(top).reshape(-1).tolist()

    def prefill(self, tokens):
        """The prompt's KV rows and the greedy token after it."""
        import ttnn

        x = self.prefill_layers(self.embed(self.ids(tokens)))
        last = len(tokens) - 1
        start = last // TILE * TILE
        rows = ttnn.slice(x, [0, 0, start, 0], [1, 1, start + TILE, self.cfg.dim])
        ttnn.deallocate(x)
        return self.top1(rows)[last - start]

    def decode(self, token, pos):
        """The greedy token after token at pos."""
        import ttnn

        x = self.embed(self.ids([token]), rows=1)
        x = self.decode_layers(x, pos)
        out = self.top1(x)[0]
        ttnn.deallocate(x)
        return out
