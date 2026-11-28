# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Xylolabs
"""Qwen3 dense (Qwen3-4B, Qwen3-8B) on one chip.

Each decoder layer: RMSNorm, attention with grouped KV heads, a per-head RMSNorm on q and k
(q_norm, k_norm) and HF-style RoPE, then RMSNorm and a SwiGLU MLP, each with its residual add.
q, k and v are one matrix ([q | k | v] along the output, whole tiles: exact). The KV cache is paged
(kv.PagedKV), bf16, one per layer.

Prefill takes the prompt from position 0 in one pass (its length padded to a multiple of 32);
decode takes one token at a position. Both run eagerly here; the embedding lookup, the RoPE rows,
the head and its argmax run on the device, and only token ids and positions cross to it."""

import math

import torch

from .. import kernels, ops
from ..kv import PagedKV
from ..rope import HfRope

TILE = 32


class Qwen3Layer:
    def __init__(self, model, i):
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
        self.kv = PagedKV(dev, c.kv_heads, c.head_dim, model.max_seq)

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
        return self._rest(x, o)

    def decode(self, x, cos, sin, pos, mcs):
        """x [1, 1, 32, dim] (row 0 the token) at the position in pos (int32 [1] on the device)."""
        import ttnn

        c = self.model.cfg
        h = ops.rows(ops.rms_norm(x, self.attn_norm, c.eps), mcs["users"])
        qkv = ops.linear(h, self.wqkv, ops.fidelity_for(self.qkv_fmt), ttnn.L1_MEMORY_CONFIG)
        ttnn.deallocate(h)
        users = mcs["users"]  # one decode row, or a verify's T rows as T users
        qkv = ttnn.reshape(qkv, (1, 1, users, qkv.shape[-1]), (1, 1, TILE, qkv.shape[-1]))
        q, k, v = ttnn.experimental.nlp_create_qkv_heads_decode(
            qkv, num_heads=c.heads, num_kv_heads=c.kv_heads, memory_config=mcs["heads"]
        )
        ttnn.deallocate(qkv)
        q = self._head_norm(q, self.q_norm, True)
        k = self._head_norm(k, self.k_norm, True)
        q = ttnn.experimental.rotary_embedding_hf(q, cos, sin, is_decode_mode=True)
        k = ttnn.experimental.rotary_embedding_hf(k, cos, sin, is_decode_mode=True)
        self.kv.update(k, v, pos, users)
        ttnn.deallocate(k)
        ttnn.deallocate(v)
        a = self.kv.attend(q, pos, 1.0 / math.sqrt(c.head_dim), ops.compute_config("hifi4"), users)
        ttnn.deallocate(q)
        a = ttnn.to_memory_config(a, mcs["sdpa_out"])
        o = ttnn.experimental.nlp_concat_heads_decode(a, num_heads=c.heads)
        ttnn.deallocate(a)
        o = ops.rows(ttnn.to_memory_config(o, ttnn.DRAM_MEMORY_CONFIG), users)
        return self._rest(x, o)

    def _rest(self, x, o):
        """The output projection, the residual adds and the MLP."""
        import ttnn

        c = self.model.cfg
        n = x.shape[-2]  # both sides of each add at x's rows
        y = ops.rows(ops.linear(o, self.wo, ops.fidelity_for(self.o_fmt)), n)
        ttnn.deallocate(o)
        s = ttnn.add(x, y)
        ttnn.deallocate(y)
        ttnn.deallocate(x)  # the layer's input, made by the step before it
        h = ops.rms_norm(s, self.mlp_norm, c.eps)
        m = ops.rows(self.mlp(h), n)
        ttnn.deallocate(h)
        out = ttnn.add(s, m)
        ttnn.deallocate(m)
        ttnn.deallocate(s)
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


def head_for(model, ex, device):
    """(the head as a ttnn matrix, None), or with the kernels on (None, kernels.Head), which takes
    its rows from the embedding table when the head is the embedding."""
    w = ex.get(ex.head_name)
    if not kernels.enabled():
        return ops.matrix(device, w, ex.head_format, kernel=False), None
    tied = ex.head_name == ex.embed_name or torch.equal(w, ex.get(ex.embed_name))
    return None, kernels.Head(device, w, model.embedding if tied else None)


class Qwen3:
    """The model on one chip. layers: the decoder layers to load (default all); the embedding, the
    final norm and the head load with head=True."""

    def __init__(self, export, device, max_seq=1024, layers=None, head=True, trace=False):
        import ttnn

        self.export, self.device, self.max_seq = export, device, max_seq
        self.trace, self._trace = trace, None
        self.cfg = c = Config(export.text_config)
        assert max_seq % TILE == 0
        self.rope = HfRope(device, c.head_dim, max_seq, c.theta)
        ids = range(c.layers) if layers is None else layers
        self.layers = {i: Qwen3Layer(self, i) for i in ids}
        self._mcs = {}
        self.mcs = self.decode_configs(1)
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
            self.final_norm = ops.norm_weight(device, ex.get(f"{ex.prefix}norm.weight"))
            self.head, self.khead = head_for(self, ex, device)

    def decode_configs(self, users=1):
        """The decode heads' height-sharded layouts for `users` rows (one decode row, or a
        verify's T rows as T users): one core each, as tt-transformers sets them. Made once per
        user count."""
        import ttnn

        if users in self._mcs:
            return self._mcs[users]
        c = self.cfg
        grid = ops.user_grid(self.device, users)

        def shard(rows):
            return ttnn.create_sharded_memory_config(
                shape=(rows, c.head_dim),
                core_grid=grid,
                strategy=ttnn.ShardStrategy.HEIGHT,
                orientation=ttnn.ShardOrientation.ROW_MAJOR,
                use_height_and_width_as_shard_shape=True,
            )

        heads = shard(TILE)
        self._mcs[users] = {
            "heads": heads,
            "sdpa_out": shard(math.ceil(c.heads / TILE) * TILE),
            "rope": heads,
            "users": users,
        }
        return self._mcs[users]

    # the device side of one token or one prompt -------------------------------------------------

    def ids(self, tokens, device=True):
        """Token ids as uint32 [1, n] (n padded to a multiple of 32 with token 0), on the device or,
        with device=False, on the host (for a copy into a traced step's input)."""
        import ttnn

        n = -(-len(tokens) // TILE) * TILE
        t = torch.zeros(1, n, dtype=torch.int32)
        t[0, : len(tokens)] = torch.tensor(tokens, dtype=torch.int32)
        return ttnn.from_torch(
            t,
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=self.device if device else None,
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

    def position(self, pos, device=True):
        """The decode position as int32 [1] (the KV cache's row) and uint32 [1, 1] (the RoPE
        gather's index), on the device or (device=False) on the host."""
        import ttnn

        dev = self.device if device else None
        p = torch.tensor([pos], dtype=torch.int32)
        cur = ttnn.from_torch(p, dtype=ttnn.int32, device=dev)
        idx = ttnn.from_torch(
            p.reshape(1, 1), dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=dev
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
        return self._decode_layers(x, cur, idx)

    def _decode_layers(self, x, cur, idx, users=1):
        mcs = self.decode_configs(users)
        cos, sin = self.rope.decode(idx, mcs["rope"])
        for i in sorted(self.layers):
            x = self.layers[i].decode(x, cos, sin, cur, mcs)
        return x

    # pcc's layer access -------------------------------------------------------------------------

    def kv_source(self, i):
        """The layer whose KV cache layer i attends over (here always i)."""
        return i

    def check_prefill(self, i, x, tokens):
        """Layer i's prefill of x [1, 1, T, dim] (the rows of tokens) from position 0."""
        cos, sin = self.rope.prefill(0, x.shape[-2])
        return self.layers[i].prefill(x, cos, sin)

    def check_decode(self, i, x, token, p):
        """Layer i's decode step of x [1, 1, 1, dim] (the row of token) at position p."""
        cur, idx = self.position(p)
        cos, sin = self.rope.decode(idx, self.mcs["rope"])
        return self.layers[i].decode(x, cos, sin, cur, self.mcs)

    def _head_top1(self, h):
        """The argmax ids of h's rows (normed) on the device: the head and argmax there."""
        import ttnn

        logits = ops.linear(h, self.head, ops.fidelity_for(self.export.head_format))
        rm = ttnn.to_layout(logits, ttnn.ROW_MAJOR_LAYOUT)
        ttnn.deallocate(logits)
        top = ttnn.argmax(rm, dim=3)
        ttnn.deallocate(rm)
        return top

    def _top1(self, x):
        """The argmax ids of x's rows on the device: the final norm, the head and argmax there."""
        import ttnn

        h = ops.rms_norm(x, self.final_norm, self.cfg.eps)
        top = self._head_top1(h)
        ttnn.deallocate(h)
        return top

    def greedy_normed(self, h, rows):
        """The greedy ids of h's first rows (normed), read back: the kernels' head when it is on
        (kernels.Head), else the head and argmax on the device. Only the ids come back."""
        import ttnn

        if self.khead is not None:
            return self.khead(h, rows)
        top = self._head_top1(h)
        out = ttnn.to_torch(top).reshape(-1)[:rows].tolist()
        ttnn.deallocate(top)
        return out

    def greedy(self, x, rows):
        """The greedy ids of x's first rows: the final norm, then greedy_normed."""
        import ttnn

        h = ops.rms_norm(x, self.final_norm, self.cfg.eps)
        out = self.greedy_normed(h, rows)
        ttnn.deallocate(h)
        return out

    def _decode_step(self, ids, cur, idx):
        """One decode step from device inputs to the argmax ids on the device."""
        x = self._decode_layers(self.embed(ids, rows=1), cur, idx)
        return self._top1(x)

    def prefill(self, tokens):
        """The prompt's KV rows and the greedy token after it."""
        import ttnn

        x = self.prefill_layers(self.embed(self.ids(tokens)))
        last = len(tokens) - 1
        # a slice of the whole tensor may be the tensor itself: x is freed only after the head
        row = ttnn.slice(x, [0, 0, last, 0], [1, 1, last + 1, self.cfg.dim])
        out = self.greedy(row, 1)[0]
        ttnn.deallocate(x)
        return out

    def decode(self, token, pos):
        """The greedy token after token at pos (eager: ttsim runs only this)."""
        import ttnn

        if self.trace:
            return self.decode_traced(token, pos)
        if self.khead is not None:
            x = self._decode_layers(self.embed(self.ids([token]), rows=1), *self.position(pos))
            out = self.greedy(x, 1)[0]
            ttnn.deallocate(x)
            return out
        top = self._decode_step(self.ids([token]), *self.position(pos))
        out = int(ttnn.to_torch(top).reshape(-1)[0])
        ttnn.deallocate(top)
        return out

    def decode_traced(self, token, pos):
        """decode() replayed from a trace: the first call runs the step eagerly on persistent
        inputs (the token id, the position), then captures the same step; later calls copy the
        new id and position into those inputs and replay. Needs a card (ttsim has no traces) and
        a device opened with a trace region. Not yet run on a card."""
        import ttnn

        if self.khead is not None:
            raise NotImplementedError("--trace with --kernels: the kernels' head is not traced yet")
        host = (self.ids([token], device=False), *self.position(pos, device=False))
        if self._trace is None:
            self._trace_in = [ttnn.to_device(h, self.device) for h in host]
            top = self._decode_step(*self._trace_in)
            out = int(ttnn.to_torch(top).reshape(-1)[0])
            ttnn.deallocate(top)
            tid = ttnn.begin_trace_capture(self.device, cq_id=0)
            self._trace_out = self._decode_step(*self._trace_in)
            ttnn.end_trace_capture(self.device, tid, cq_id=0)
            self._trace = tid
            return out
        for h, d in zip(host, self._trace_in):
            ttnn.copy_host_to_device_tensor(h, d)
        ttnn.execute_trace(self.device, self._trace, cq_id=0, blocking=False)
        return int(ttnn.to_torch(self._trace_out).reshape(-1)[0])

    def verify(self, tokens, pos):
        """The greedy token after each of tokens (T of them, at positions pos .. pos + T - 1) as
        one decode step of T rows; the DeltaNet layers keep every row's state until commit().
        Returns (the T ids, the last layer's rows [1, 1, T, dim] on the device)."""
        import ttnn

        T = len(tokens)
        p = torch.arange(pos, pos + T, dtype=torch.int32)
        cur = ttnn.from_torch(p, dtype=ttnn.int32, device=self.device)
        idx = ttnn.from_torch(
            p.reshape(1, T), dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=self.device
        )
        x = self._decode_layers(self.embed(self.ids(tokens), rows=T), cur, idx, T)
        return self.greedy(x, T), x

    def commit(self, a):
        """Keep the first a rows of the last verify: nothing to do here (the KV rows past them
        are written over later)."""

    def release_trace(self):
        import ttnn

        if self._trace is not None:
            ttnn.release_trace(self.device, self._trace)
        self._trace = None
