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


@pytest.mark.parametrize(
    "extra",
    [
        ["--spec", "4", "--mtp"],
        ["--spec", "4", "--verify", "4"],
        ["--verify", "4", "--drift"],
        ["--drift", "--layers", "0"],
        ["--mtp", "--layers", "0"],
    ],
)
def test_pcc_rejects_combined_primary_checks(extra):
    with pytest.raises(SystemExit, match="runs one of"):
        cli.main(["pcc", "no-such-export", *extra])


def test_pcc_keeps_layers_as_a_selector(monkeypatch):
    called = []
    monkeypatch.setattr(cli, "cmd_pcc", lambda a: called.append(a) or 0)
    for extra in (["--spec", "4", "--layers", "0"], ["--verify", "4", "--layers", "0"]):
        assert cli.main(["pcc", "x", *extra]) == 0
    assert len(called) == 2


def test_pcc_sizes_the_cache_without_an_unused_k(monkeypatch, fake_ttnn):
    from tensix_runner import device, export, models, pcc

    class Stop(Exception):
        pass

    seen = {}

    def load(ex, dev, **kw):
        seen.update(kw)
        raise Stop

    ex = SimpleNamespace(close=lambda: None, text_config={"num_hidden_layers": 2}, formats=[])
    monkeypatch.setattr(export, "Export", lambda path: ex)
    monkeypatch.setattr(device, "open_device", lambda *a, **k: object())
    monkeypatch.setattr(device, "info", lambda dev: "fake")
    monkeypatch.setattr(models, "load", load)
    monkeypatch.setattr(pcc, "prompt_tokens", lambda ex, n: [1] * n)
    with pytest.raises(Stop):
        cli.main(["pcc", "x", "--greedy", "1", "--k", "1000000"])
    assert seen["max_seq"] == 256, seen




# device ----------------------------------------------------------------------------------------


def _ttnn():
    return pytest.importorskip("ttnn")


def test_decode_sharding_follows_the_device_grid(device):
    import ttnn

    from tensix_runner.models.qwen3 import Qwen3

    grid = device.compute_with_storage_grid_size()
    users = min(32, grid.x * grid.y)
    me = SimpleNamespace(_mcs={}, device=device, cfg=SimpleNamespace(head_dim=128, heads=32))
    configs = Qwen3.decode_configs(me, users)  # 32 users reach past a grid of fewer columns
    assert configs["users"] == users
    import torch

    t = ttnn.from_torch(
        torch.zeros(1, 1, users * 32, 128),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    sharded = ttnn.to_memory_config(t, configs["heads"])  # fails on cores outside the grid
    assert sharded.is_sharded()


def test_user_grid_is_row_major_over_the_device(device):
    from tensix_runner import ops

    grid = device.compute_with_storage_grid_size()
    for users in (1, grid.x - 1, grid.x, grid.x + 3, grid.x * grid.y):
        assert ops.user_grid(device, users).num_cores() == users, users
    ranges = ops.user_grid(device, grid.x + 3).ranges()
    ends = sorted((r.start.x, r.start.y, r.end.x, r.end.y) for r in ranges)
    assert ends == [(0, 0, grid.x - 1, 0), (0, 1, 2, 1)], ends  # a whole row, then 3 cores
    for users in (0, grid.x * grid.y + 1):
        with pytest.raises(ValueError, match="do not fit"):
            ops.user_grid(device, users)


def test_compute_config_is_made_once():
    _ttnn()
    from tensix_runner import ops

    assert ops.compute_config("hifi4") is ops.compute_config("hifi4")
    assert ops.compute_config("hifi2") is not ops.compute_config("hifi4")
