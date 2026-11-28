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


class SpecModel:
    """A model whose verify returns the drafts it was given plus one more token, all accepted."""

    max_seq = 256

    def __init__(self, first=5):
        self.first, self.commits = first, []

    def prefill_hidden(self, tokens):
        return self.first, MagicMock()

    def verify(self, rows, p):
        return [100 + i for i in range(len(rows))], MagicMock()

    def commit(self, n):
        self.commits.append(n)


class SpecMtp:
    def __init__(self):
        self.calls = 0

    def draft(self, x, toks, p, k):
        self.calls += 1
        return [100 + i for i in range(k)]


def test_speculate_commits_only_the_returned_rows(fake_ttnn):
    model, mtp = SpecModel(), SpecMtp()
    out, _, _, kept = g.speculate(model, mtp, [1, 2, 3], max_new=2, k=3)
    assert out == [5, 100]
    assert model.commits == [1] and kept == [0]  # not 4 rows committed and 3 drafts kept


def test_speculate_commits_up_to_a_stop_token(fake_ttnn):
    model, mtp = SpecModel(), SpecMtp()
    out, _, _, kept = g.speculate(model, mtp, [1, 2, 3], max_new=10, k=3, stop={101})
    assert out == [5, 100, 101]
    assert model.commits == [2] and kept == [1]


def test_speculate_drafts_nothing_when_prefill_completes(fake_ttnn):
    model, mtp = SpecModel(), SpecMtp()
    seen = []
    out, _, _, kept = g.speculate(model, mtp, [1, 2, 3], max_new=1, k=3, on_token=seen.append)
    assert (out, kept, seen, mtp.calls, model.commits) == ([5], [], [5], 0, [])
    out, _, _, kept = g.speculate(model, mtp, [1, 2, 3], max_new=9, k=3, stop={5})
    assert (out, kept, mtp.calls) == ([5], [], 0)


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






