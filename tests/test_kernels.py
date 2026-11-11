# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Xylolabs
"""The optional tensix-kernels path: a checkout is found or refused on the host; on the device
(TENSIX_KERNELS a tensix-kernels checkout), the dense reader's matmul against ttnn.linear on a
matrix already on its BFP grid, at 1 and 32 rows and with an out dimension that does not divide
over the banks."""

import pytest
import torch

from tensix_runner import kernels


def test_enable_refuses_other_directories(tmp_path):
    with pytest.raises(FileNotFoundError):
        kernels.enable(str(tmp_path))


def test_enable_is_off_without_a_path(monkeypatch):
    monkeypatch.delenv("TENSIX_KERNELS", raising=False)
    assert kernels.enable(None) is False


@pytest.mark.device
@pytest.mark.parametrize("fmt, out_dim", [("bfp4", 1024), ("bfp8", 1024), ("bf16", 96)])
@pytest.mark.parametrize("rows", [1, 32])
def test_dense_reader_matches_linear(device, monkeypatch, fmt, out_dim, rows):
    import os

    import ttnn

    from tensix_runner import ops
    from tensix_runner.bfp import bfp
    from tensix_runner.pcc import pcc

    if not os.environ.get("TENSIX_KERNELS"):
        pytest.skip("TENSIX_KERNELS names no tensix-kernels checkout")
    ops.set_arch(device)
    g = torch.Generator().manual_seed(rows)
    w = torch.randn(out_dim, 512, generator=g) * 0.05
    if fmt != "bf16":  # on the format's grid, as an export's matrix is
        w = bfp(w.T.contiguous(), {"bfp4": 3, "bfp8": 7}[fmt]).T.contiguous()
    w = w.to(torch.bfloat16)
    x = ttnn.from_torch(
        (torch.randn(1, 1, 32, 512, generator=g) * 0.5).to(torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
    )
    plain = ops.matrix(device, w, fmt, kernel=False)
    want = ttnn.to_torch(ops.linear(x, plain)).float()[0, 0, :rows]
    monkeypatch.setitem(kernels._STATE, "dr", None)  # restored after the test: the path stays off
    monkeypatch.setitem(kernels._STATE, "path", None)
    assert kernels.enable()
    dense = ops.matrix(device, w, fmt)
    assert isinstance(dense, kernels.Weight)
    got = ttnn.to_torch(ops.linear(x, dense)).float().reshape(-1, out_dim)[:rows]
    assert got.shape == want.shape
    if device.arch() == ttnn.device.Arch.BLACKHOLE:  # both accumulate in fp32 at HiFi4
        assert torch.equal(got, want)
    else:  # ttnn.linear accumulates in bf16 on the Wormhole here
        assert pcc(got, want) > 0.9999
