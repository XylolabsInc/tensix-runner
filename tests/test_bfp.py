# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Xylolabs
"""The torch grid check: bfp() on hand-computed blocks and on_grid() on rounded and raw weights."""

import torch

from tensix_runner.bfp import bfp, on_grid


def test_block_shares_the_largest_exponent():
    # one block: 8.0 sets the exponent (2^3); bfloat4_b keeps 3 mantissa bits, a step of 2^(3-2) = 2
    x = torch.tensor([[8.0, 1.0, 3.0, 5.0, -7.0, 0.0, 2.9, 1.1] + [0.0] * 8])
    got = bfp(x, 3)
    want = torch.tensor([[8.0, 0.0, 4.0, 4.0, -8.0, 0.0, 2.0, 2.0] + [0.0] * 8])
    assert torch.equal(got, want)


def test_round_half_to_even_and_clamp():
    # exponent 2^0, bfloat8_b: a step of 2^-6; 1 + 2^-7 is a tie that rounds to the even 1.0
    x = torch.tensor([[1.0 + 2**-7, 1.0 + 3 * 2**-7] + [0.0] * 14])
    got = bfp(x, 7)
    assert got[0, 0] == 1.0 and got[0, 1] == 1.0 + 2 * 2**-6
    # the largest value just under the next power of two clamps at the top code instead of overflowing
    y = torch.tensor([[1.9999] + [0.0] * 15])
    assert bfp(y, 3)[0, 0] == 1.75


def test_on_grid():
    g = torch.Generator().manual_seed(1)
    w = torch.randn(48, 64, generator=g)
    for fmt, bits in (("bfp4", 3), ("bfp8", 7)):
        r = bfp(w.T.contiguous(), bits).T.contiguous()
        assert on_grid(r, fmt)
        assert not on_grid(w, fmt)
    assert on_grid(w.bfloat16(), "bf16")
    # a bfloat4_b value is also on the bfloat8_b grid of the same block
    assert on_grid(bfp(w.T.contiguous(), 3).T.contiguous(), "bfp8")
