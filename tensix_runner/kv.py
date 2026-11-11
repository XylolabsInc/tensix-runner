# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Xylolabs
"""A layer's paged KV cache for one sequence.

K and V live in blocks of BLOCK positions, [blocks, kv_heads, BLOCK, head_dim], with a page table
that maps the sequence's block i to block i. Paging lets several rows of one sequence act as
separate "users" of the same pages: a verify of T draft rows is a decode step of T users whose
page-table rows are all the sequence's and whose positions are p, p + 1, ..., p + T - 1, so the
decode attention gives each row exactly its causal context."""

import torch

BLOCK = 32


class PagedKV:
    def __init__(self, device, kv_heads, head_dim, max_seq):
        import ttnn

        assert max_seq % BLOCK == 0, f"max_seq {max_seq} is not a multiple of {BLOCK}"
        self.device, self.blocks = device, max_seq // BLOCK
        self.head_dim, self._one = head_dim, None
        shape = [self.blocks, kv_heads, BLOCK, head_dim]
        kw = {"dtype": ttnn.bfloat16, "layout": ttnn.TILE_LAYOUT, "device": device}
        self.k = ttnn.zeros(shape, memory_config=ttnn.DRAM_MEMORY_CONFIG, **kw)
        self.v = ttnn.zeros(shape, memory_config=ttnn.DRAM_MEMORY_CONFIG, **kw)
        self._tables = {}

    def table(self, users=1):
        """The page table [users, blocks] (int32, row major, on the device): every row the
        sequence's blocks in order. Made once per user count."""
        import ttnn

        if users not in self._tables:
            t = torch.arange(self.blocks, dtype=torch.int32).repeat(users, 1)
            self._tables[users] = ttnn.from_torch(
                t, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT, device=self.device
            )
        return self._tables[users]

    def fill(self, k, v):
        """A prefill's K and V [1, kv_heads, T, head_dim] from position 0."""
        import ttnn

        ttnn.experimental.paged_fill_cache(self.k, k, self.table(), batch_idx=0)
        ttnn.experimental.paged_fill_cache(self.v, v, self.table(), batch_idx=0)

    def update(self, k, v, pos, users=1):
        """Decode rows [1, users, kv_heads (32), head_dim], height sharded one user a core, written
        at the positions in pos (int32 [users]).

        paged_update_cache rewrites, for each user and on that user's core, the whole tile of the
        block its row falls in, all users at once: users whose rows share a block keep only one of
        their rows (on ttsim, the last user's). A verify's rows are one sequence's consecutive
        positions, so they are written one user at a time."""
        import ttnn

        t = self.table(1)
        if users == 1:
            ttnn.experimental.paged_update_cache(self.k, k, update_idxs_tensor=pos, page_table=t)
            ttnn.experimental.paged_update_cache(self.v, v, update_idxs_tensor=pos, page_table=t)
            return
        if self._one is None:  # one user's row on one core
            core = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))})
            self._one = ttnn.create_sharded_memory_config(
                shape=(32, self.head_dim),
                core_grid=core,
                strategy=ttnn.ShardStrategy.HEIGHT,
                orientation=ttnn.ShardOrientation.ROW_MAJOR,
                use_height_and_width_as_shard_shape=True,
            )
        idx = [ttnn.slice(pos, [u], [u + 1]) for u in range(users)]
        for rows, cache in ((k, self.k), (v, self.v)):
            flat = ttnn.sharded_to_interleaved(rows, ttnn.L1_MEMORY_CONFIG)
            h, d = flat.shape[2], flat.shape[3]
            for u in range(users):
                one = ttnn.interleaved_to_sharded(
                    ttnn.slice(flat, [0, u, 0, 0], [1, u + 1, h, d]), self._one
                )
                ttnn.experimental.paged_update_cache(
                    cache, one, update_idxs_tensor=idx[u], page_table=t
                )
                ttnn.deallocate(one)
            ttnn.deallocate(flat)
        for i in idx:
            ttnn.deallocate(i)

    def attend(self, q, pos, scale, ckc, users=1):
        """Decode attention of q [1, users, heads (32), head_dim] over each user's positions up to
        and including its own in pos."""
        import ttnn

        return ttnn.transformer.paged_scaled_dot_product_attention_decode(
            q,
            self.k,
            self.v,
            page_table_tensor=self.table(users),
            cur_pos_tensor=pos,
            scale=scale,
            compute_kernel_config=ckc,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
