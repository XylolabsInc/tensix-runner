# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Xylolabs
"""Rotary position embedding in Hugging Face's layout (rotate_half: the first and second halves of
each head pair up), for ttnn.experimental.rotary_embedding_hf.

The cos and sin tables are computed once at load and kept on the device: a tile copy for prefill
(rows sliced on the device) and a row-major copy that decode gathers rows from with ttnn.embedding,
by the position ids on the device. Per token only the position id crosses to the device."""

import torch


def inv_freq(head_dim, theta, rotary_dim=None):
    """HF's default inverse frequencies: theta^(-2i/d) over the rotated dimensions."""
    d = rotary_dim or head_dim
    return 1.0 / (theta ** (torch.arange(0, d, 2, dtype=torch.int64).float() / d))


def tables(head_dim, max_seq, theta, inv=None):
    """cos and sin [max_seq, head_dim] in HF's layout, float32 (computed as transformers does), from
    the default inverse frequencies or the head_dim / 2 given in inv."""
    f = inv_freq(head_dim, theta) if inv is None else inv
    t = torch.arange(max_seq, dtype=torch.int64).float()
    ang = torch.outer(t, f)
    emb = torch.cat([ang, ang], dim=-1)
    return emb.cos(), emb.sin()


class HfRope:
    def __init__(self, device, head_dim, max_seq, theta, inv=None):
        import ttnn

        self.device, self.head_dim, self.max_seq = device, head_dim, max_seq
        cos, sin = tables(head_dim, max_seq, theta, inv)
        dram = ttnn.DRAM_MEMORY_CONFIG

        def put(t, layout, shape):
            return ttnn.as_tensor(
                t.to(torch.bfloat16).reshape(shape),
                dtype=ttnn.bfloat16,
                layout=layout,
                device=device,
                memory_config=dram,
            )

        self.cos_tile = put(cos, ttnn.TILE_LAYOUT, (1, 1, max_seq, head_dim))
        self.sin_tile = put(sin, ttnn.TILE_LAYOUT, (1, 1, max_seq, head_dim))
        self.cos_rows = put(cos, ttnn.ROW_MAJOR_LAYOUT, (max_seq, head_dim))
        self.sin_rows = put(sin, ttnn.ROW_MAJOR_LAYOUT, (max_seq, head_dim))

    def prefill(self, start, n):
        """cos and sin [1, 1, n, head_dim] for positions start .. start + n - 1 (tile aligned)."""
        import ttnn

        assert start % 32 == 0 and n % 32 == 0 and start + n <= self.max_seq, (start, n)
        end = [1, 1, start + n, self.head_dim]
        return (
            ttnn.slice(self.cos_tile, [0, 0, start, 0], end),
            ttnn.slice(self.sin_tile, [0, 0, start, 0], end),
        )

    def decode(self, pos, shard_mc):
        """cos and sin [1, batch, 1, head_dim] for the positions pos (uint32 [1, batch] on the
        device), height sharded like the decode heads (shard_mc)."""
        import ttnn

        out = []
        for table in (self.cos_rows, self.sin_rows):
            r = ttnn.embedding(pos, table, layout=ttnn.TILE_LAYOUT)  # [1, batch, head_dim]
            r = ttnn.transpose(ttnn.unsqueeze_to_4D(r), 1, 2)  # [1, batch, 1 (32), head_dim]
            out.append(ttnn.interleaved_to_sharded(r, shard_mc))
        return out
