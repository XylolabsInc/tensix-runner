# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Xylolabs
"""The paged KV cache on the device: a verify's rows (several users at consecutive positions of
one sequence, inside one block) all land, and a decode row lands where pos says."""

import pytest
import torch

pytestmark = pytest.mark.device


def rows(device, values, kv_heads, head_dim):
    """[1, users, kv_heads, head_dim] whose user u holds values[u], height sharded one user a core."""
    import ttnn

    users = len(values)
    x = torch.zeros(1, users, 32, head_dim)
    for u, val in enumerate(values):
        x[0, u, :kv_heads] = val
    t = ttnn.from_torch(
        x.to(torch.bfloat16), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
    )
    t = ttnn.reshape(t, (1, users, kv_heads, head_dim), (1, users, 32, head_dim))
    grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(users - 1, 0))})
    mc = ttnn.create_sharded_memory_config(
        shape=(32, head_dim),
        core_grid=grid,
        strategy=ttnn.ShardStrategy.HEIGHT,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )
    return ttnn.interleaved_to_sharded(t, mc)


def cached(kv):
    """The K cache's first channel of the first KV head, per position."""
    import ttnn

    c = ttnn.to_torch(kv.k).float()  # [blocks, kv_heads, 32, head_dim]
    return c[:, 0, :, 0].reshape(-1)


@pytest.mark.parametrize("start", [32, 46, 62])  # inside one block, and across a block's end
def test_verify_rows_all_land(device, start):
    import ttnn

    from tensix_runner.kv import PagedKV

    kv = PagedKV(device, 1, 64, 128)
    values = [1.0, 2.0, 3.0]
    x = rows(device, values, 1, 64)
    pos = ttnn.from_torch(
        torch.arange(start, start + 3, dtype=torch.int32), dtype=ttnn.int32, device=device
    )
    kv.update(x, x, pos, users=3)
    assert cached(kv)[start : start + 3].tolist() == values


def test_decode_row(device):
    import ttnn

    from tensix_runner.kv import PagedKV

    kv = PagedKV(device, 1, 64, 128)
    x = rows(device, [5.0], 1, 64)
    pos = ttnn.from_torch(torch.tensor([77], dtype=torch.int32), dtype=ttnn.int32, device=device)
    kv.update(x, x, pos)
    got = cached(kv)
    assert got[77] == 5.0 and got.abs().sum() == 5.0
