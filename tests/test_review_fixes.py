# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Xylolabs
"""Regression tests for the correctness and host-overhead review fixes of 2026-11-28: Qwen3.5
prefill normalization, speculation commits, decode sharding over the device's grid, pcc's argument
handling and cache size, bench warmup, the token count check, the incremental stream and the
cached compute configs. Most run on the host with stand-ins for the device; the sharding and
compute-config tests use ttnn (and ttsim) when it is installed."""

import sys
from types import SimpleNamespace
from typing import ClassVar
from unittest.mock import MagicMock

import pytest

from tensix_runner import cli
from tensix_runner import generate as g


@pytest.fixture(autouse=True)
def no_kernels_env(monkeypatch):
    monkeypatch.delenv("TENSIX_KERNELS", raising=False)  # the argument limits differ with it


@pytest.fixture
def fake_ttnn(monkeypatch):
    t = MagicMock(name="ttnn")
    monkeypatch.setitem(sys.modules, "ttnn", t)
    return t


# Qwen3.5 prefill -------------------------------------------------------------------------------


def test_qwen35_prefill_normalizes_q_and_k(fake_ttnn, monkeypatch):
    from tensix_runner import ops
    from tensix_runner.models.qwen35 import GatedDeltaNet

    monkeypatch.setattr(ops, "linear", lambda *a, **k: MagicMock())
    monkeypatch.setattr(ops, "fidelity_for", lambda f: "hifi4")
    fake_ttnn.transformer.chunk_gated_delta_rule.return_value = (MagicMock(), MagicMock())
    calls = []
    cfg = SimpleNamespace(conv=4, v_heads=2, k_heads=1, v_dim=8)
    me = SimpleNamespace(
        model=SimpleNamespace(cfg=cfg),
        kd=8,
        vd=16,
        wqkv=None,
        wz=None,
        qkv_fmt="bfp4",
        z_fmt="bfp4",
        state=None,
        conv_dim=32,
        conv_state=[MagicMock() for _ in range(3)],
        _gates=lambda h, T: (MagicMock(), MagicMock()),
        _conv_prefill=lambda rm: MagicMock(),
        _out=lambda o, z: "out",
        _l2=lambda x, heads, rows=1: calls.append((heads, rows)) or x,
    )
    h = MagicMock()
    h.shape = (1, 1, 32, 64)
    assert GatedDeltaNet.prefill(me, h, None, None, 32) == "out"
    assert calls == [(1, 32), (1, 32)], "q and k are normalized per head, as in decode and verify"


# speculation -----------------------------------------------------------------------------------












# token counts ----------------------------------------------------------------------------------


@pytest.mark.parametrize("bad", [1.5, True, None, -1, "3"])
def test_check_count_requires_a_nonnegative_integer(bad):
    with pytest.raises(ValueError, match="max_new"):
        g.check_count(bad)
    g.check_count(0)
    g.check_count(7)


# the stream ------------------------------------------------------------------------------------








# pcc and bench ---------------------------------------------------------------------------------










# device ----------------------------------------------------------------------------------------


def _ttnn():
    return pytest.importorskip("ttnn")






