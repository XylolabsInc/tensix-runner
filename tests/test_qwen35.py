# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Xylolabs
"""Qwen3.5's partial RoPE through the channel permutation, on the host: rotate_half over the whole
permuted head with tables() equals transformers' rotation of the first rotary channels, and the
permutation moves whole tiles of 32 channels (so permuted BFP weights stay exact)."""

import sys
import types

import pytest
import torch

from tensix_runner.models.qwen35 import Qwen35, head_perm, permute_heads, tables
from tensix_runner.rope import inv_freq


def rotate_half(x):
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def hf_partial(x, pos, d, r, theta):
    """transformers' apply_rotary_pos_emb for one head: rotate the first r channels only."""
    f = inv_freq(d, theta, r)
    ang = torch.outer(pos.float(), f)
    emb = torch.cat([ang, ang], -1)
    cos, sin = emb.cos(), emb.sin()
    rot, rest = x[..., :r], x[..., r:]
    return torch.cat([rot * cos + rotate_half(rot) * sin, rest], -1)


def test_partial_rope_by_permutation():
    d, r, theta = 256, 64, 1e7
    perm = head_perm(d, r)
    assert sorted(perm) == list(range(d))
    pos = torch.tensor([0, 1, 7, 300, 4095])
    x = torch.randn(len(pos), d, dtype=torch.float64)
    want = hf_partial(x.float(), pos, d, r, theta)
    cos, sin = tables(d, r, theta, 4096)
    xp = x.float()[:, perm]
    got_p = xp * cos[pos] + rotate_half(xp) * sin[pos]
    got = torch.empty_like(got_p)
    got[:, perm] = got_p  # back to the checkpoint's order
    assert torch.allclose(got, want, atol=1e-5)


def test_dot_products_unchanged():
    d, r = 256, 64
    perm = head_perm(d, r)
    q, k = torch.randn(5, d), torch.randn(7, d)
    assert torch.allclose(q[:, perm] @ k[:, perm].T, q @ k.T, atol=1e-5)


def test_permutation_moves_whole_tiles():
    d, r = 256, 64
    perm = head_perm(d, r)
    for t in range(0, d, 32):
        block = perm[t : t + 32]
        assert block[0] % 32 == 0 and block == list(range(block[0], block[0] + 32)), (t, block[:3])
    w = torch.arange(2 * d * 3, dtype=torch.float32).reshape(2 * d, 3)
    p = permute_heads(w, d, perm)
    assert torch.equal(p[:d], w[:d][perm]) and torch.equal(p[d:], w[d:][perm])


def test_traced_decode_is_rejected_before_device_setup(monkeypatch):
    monkeypatch.delitem(sys.modules, "ttnn", raising=False)
    with pytest.raises(NotImplementedError, match="traced decode"):
        Qwen35(None, None, trace=True)


def test_eager_constructor_reaches_device_check(monkeypatch):
    ttnn = types.ModuleType("ttnn")
    ttnn.device = types.SimpleNamespace(Arch=types.SimpleNamespace(BLACKHOLE="blackhole"))
    monkeypatch.setitem(sys.modules, "ttnn", ttnn)
    device = types.SimpleNamespace(arch=lambda: "wormhole")
    with pytest.raises(NotImplementedError, match="Blackhole"):
        Qwen35(None, device, trace=False)
