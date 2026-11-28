# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Xylolabs
"""compute_config's matmul accumulation: bf16 on ttsim's Wormhole only, fp32 everywhere else (needs the
ttnn package, no device)."""

import pytest

ttnn = pytest.importorskip("ttnn")

from tensix_runner import ops


class Device:
    def __init__(self, arch):
        self._arch = arch

    def arch(self):
        return self._arch


@pytest.mark.parametrize(
    "arch, simulated, fp32",
    [
        ("WORMHOLE_B0", True, False),
        ("WORMHOLE_B0", False, True),
        ("BLACKHOLE", True, True),
        ("BLACKHOLE", False, True),
    ],
)
def test_matmul_accumulation(monkeypatch, arch, simulated, fp32):
    if simulated:
        monkeypatch.setenv("TT_METAL_SIMULATOR", "/path/to/libttsim.so")
    else:
        monkeypatch.delenv("TT_METAL_SIMULATOR", raising=False)
    dev = Device(getattr(ttnn.device.Arch, arch))
    assert ops.compute_config("hifi4", matmul=True, device=dev).fp32_dest_acc_en is fp32
    assert ops.compute_config("hifi4", matmul=True).fp32_dest_acc_en is True
    assert ops.compute_config("hifi4", device=dev).fp32_dest_acc_en is True
