# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Xylolabs
"""One Tenstorrent chip through ttnn, and what the runner needs to know about it, read from the
device itself: the architecture, the worker grid, the DRAM banks and the L1 per core. Nothing here
assumes a particular card (n150, n300's first chip, p100, p150, or ttsim's simulated chips)."""

from dataclasses import dataclass


@dataclass(frozen=True)
class DeviceInfo:
    arch: str  # "wormhole_b0" or "blackhole"
    grid: tuple  # the compute grid (x, y)
    dram_banks: int
    l1_bytes: int  # a worker core's L1 not reserved by the firmware
    simulator: bool

    def __str__(self):
        sim = " (simulated)" if self.simulator else ""
        return (
            f"{self.arch}{sim}: compute grid {self.grid[0]} x {self.grid[1]}, "
            f"{self.dram_banks} DRAM banks, {self.l1_bytes // 1024} KiB of free L1 a core"
        )


def open_device(device_id=0, l1_small_size=24576, trace_region_size=0):
    """ttnn.open_device with the small L1 region convolutions and some kernels use, and a trace
    region when traces are wanted (ttsim runs without traces: slow dispatch only)."""
    import ttnn

    return ttnn.open_device(
        device_id=device_id, l1_small_size=l1_small_size, trace_region_size=trace_region_size
    )


def info(device):
    """The DeviceInfo of an open device."""
    import os

    import ttnn

    grid = device.compute_with_storage_grid_size()
    arch = device.arch()
    name = {ttnn.device.Arch.WORMHOLE_B0: "wormhole_b0", ttnn.device.Arch.BLACKHOLE: "blackhole"}
    return DeviceInfo(
        arch=name.get(arch, str(arch).split(".")[-1].lower()),
        grid=(grid.x, grid.y),
        dram_banks=device.dram_grid_size().x,
        l1_bytes=_l1_unreserved(device),
        simulator=bool(os.environ.get("TT_METAL_SIMULATOR")),
    )


def _l1_unreserved(device):
    """A worker core's L1 not reserved by the firmware (ttnn.device.get_max_worker_l1_unreserved_size)."""
    import ttnn

    f = ttnn.device.get_max_worker_l1_unreserved_size
    try:
        return int(f(device))
    except TypeError:  # the binding takes no device in some releases
        return int(f())
