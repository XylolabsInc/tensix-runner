# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Xylolabs
"""The greedy check's bookkeeping on the host: bfloat16's spacing and the tie rule."""

import torch

from tensix_runner import pcc


def test_bf16_ulp():
    assert pcc.bf16_ulp(15.6) == 0.0625
    assert pcc.bf16_ulp(-1.0) == 2**-7
    assert pcc.bf16_ulp(100.0) == 0.5


def test_compare_greedy():
    logits = [torch.zeros(10, dtype=torch.float64) for _ in range(3)]
    logits[1][4], logits[1][5] = 15.634, 15.585  # 0.049 apart: within one step (0.0625)
    logits[2][6], logits[2][7] = 15.6, 15.0
    same, ties, wrong = pcc.compare_greedy([1, 5, 7], [1, 4, 6], logits)
    assert same == 1
    assert [t[:3] for t in ties] == [(1, 5, 4)]
    assert [w[:3] for w in wrong] == [(2, 7, 6)]
