# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Xylolabs
"""Weights packed for TT-NN exactly as the export stores them.

TT-NN's matmul reads a weight as [in, out], the transpose of a PyTorch Linear weight. pack()
transposes a [out, in] matrix and packs it at its export format with ttnn.as_tensor. The export's
values already sit on that format's grid, so packing moves none of them; check() proves it for a
matrix, on the host and optionally through device DRAM."""

import torch


def dtype(fmt):
    """The ttnn dtype of a format name ("bf16", "bfp8", "bfp4")."""
    import ttnn

    return {"bf16": ttnn.bfloat16, "bfp8": ttnn.bfloat8_b, "bfp4": ttnn.bfloat4_b}[fmt]


def pack(w, fmt):
    """A PyTorch [out, in] weight as a host ttnn tensor [in, out] in tile layout at fmt."""
    import ttnn

    return ttnn.as_tensor(w.T.contiguous(), dtype=dtype(fmt), layout=ttnn.TILE_LAYOUT)


def unpack(t):
    """A host or device ttnn tensor as a float32 torch tensor."""
    import ttnn

    return ttnn.to_torch(t).float()


def check(w, fmt, device=None):
    """Whether w packs at fmt without moving a value: pack, (with device) write to DRAM and read
    back, unpack, and compare bit for bit with the stored values in the [in, out] layout."""
    import ttnn

    t = pack(w, fmt)
    if device is not None:
        t = ttnn.from_device(ttnn.to_device(t, device, memory_config=ttnn.DRAM_MEMORY_CONFIG))
    return torch.equal(unpack(t), w.T.float())
