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
        """Decode rows [1, users, kv_heads (32), head_dim] written at the positions in pos."""
        import ttnn

        t = self.table(users)
        ttnn.experimental.paged_update_cache(self.k, k, update_idxs_tensor=pos, page_table=t)
        ttnn.experimental.paged_update_cache(self.v, v, update_idxs_tensor=pos, page_table=t)

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
