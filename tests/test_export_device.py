# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Xylolabs
"""A real export (TENSIX_RUNNER_EXPORT, a downloaded directory) through ttnn's packer and a device's
DRAM, layers 0 and the last (TENSIX_RUNNER_LAYERS overrides them; "all" checks every layer)."""

import os

import pytest

from tensix_runner import weights
from tensix_runner.export import Export

pytestmark = [pytest.mark.device, pytest.mark.export]


def test_export_packs_exactly(export_dir, device):
    ex = Export(export_dir)
    last = max(layer for layer, _ in ex.formats)
    want = os.environ.get("TENSIX_RUNNER_LAYERS", f"0,{last}")
    layers = None if want == "all" else {int(x) for x in want.split(",")}
    moved = [
        f"layer {layer} {proj} ({fmt})"
        for layer, proj, fmt, name in ex.matrices(layers)
        if fmt != "bf16" and not weights.check(ex.get(name), fmt, device)
    ]
    assert moved == []
