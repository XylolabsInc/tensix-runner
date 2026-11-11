# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Xylolabs
"""The device side (a card or ttsim; skipped without one): what the device reports, and that values
on the BFP4 and BFP8 grids pack and travel through DRAM unchanged while off-grid values do not."""

import pytest
import torch

from tensix_runner import weights
from tensix_runner.bfp import bfp
from tensix_runner.device import info

pytestmark = pytest.mark.device


def test_info(device):
    d = info(device)
    assert d.arch in ("wormhole_b0", "blackhole")
    assert d.grid[0] > 0 and d.grid[1] > 0
    assert d.dram_banks in (7, 8, 12), d  # p100, p150 or a ttsim Blackhole; Wormhole
    assert d.l1_bytes > 1 << 20


@pytest.mark.parametrize("fmt, bits", [("bfp4", 3), ("bfp8", 7)])
def test_pack_exact(device, fmt, bits):
    g = torch.Generator().manual_seed(2)
    # [out, in] with scales from 1e-3 to 10 across the outputs, rounded onto the grid in [in, out]
    w = torch.randn(256, 128, generator=g) * torch.logspace(-3, 1, 256)[:, None]
    on = bfp(w.T.contiguous(), bits).T.contiguous().to(torch.bfloat16)
    assert weights.check(on, fmt)
    assert weights.check(on, fmt, device)
    assert not weights.check(w.to(torch.bfloat16), fmt)
