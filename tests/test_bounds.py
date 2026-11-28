# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Xylolabs
"""Counts and lengths are checked before the runner runs: speculation's k, the greedy check's
lengths and allocation, the benchmark's lengths, zero and negative generation counts, and a pcc
command line that checks nothing. Stub models, heads and exports; no device is opened."""

import sys
import types
from pathlib import Path

import pytest
import torch

from tensix_runner import cli, kernels, pcc
from tensix_runner import generate as g

EOS = 7


class Stub:
    """A model that counts its calls; decodes position + 1, or EOS at one step."""

    def __init__(self, max_seq=64, eos_at=None, khead=None):
        self.max_seq, self.eos_at, self.khead = max_seq, eos_at, khead
        self.calls, self.verify_rows = [], []

    def prefill(self, tokens):
        self.calls.append(("prefill", len(tokens)))
        return 5

    def decode(self, token, pos):
        self.calls.append(("decode", pos))
        return EOS if pos == self.eos_at else 5

    def prefill_hidden(self, tokens):
        self.calls.append(("prefill_hidden", len(tokens)))
        return 5, None

    def verify(self, rows, pos):
        self.verify_rows.append(len(rows))
        raise RuntimeError("stop after the first verify")


class StubMtp:
    def draft(self, x, tokens, pos, k):
        return [5] * k


@pytest.fixture
def ttnn_stub(monkeypatch):
    """ttnn and the modules that import it replaced by stubs for one test (the module of the
    speculation loop imports ttnn when it runs)."""
    mod = types.ModuleType("ttnn")
    mod.deallocate = lambda x: None
    mod.close_device = lambda d: None
    monkeypatch.setitem(sys.modules, "ttnn", mod)
    return mod


# M1


@pytest.mark.parametrize("k", [0, -1, 1.5, True, None])
def test_k_must_be_an_integer_of_at_least_one(ttnn_stub, k):
    m = Stub()
    with pytest.raises(ValueError, match="k is"):
        g.speculate(m, StubMtp(), [1, 2, 3], 4, k=k)
    assert m.calls == []


def test_k_error_names_the_value():
    with pytest.raises(ValueError, match="-1"):
        g.check_k(-1)


def test_k_is_limited_by_the_kernels_head(ttnn_stub):
    m = Stub(khead=object())
    with pytest.raises(ValueError, match=r"at most 8 rows.*largest k is 7"):
        g.speculate(m, StubMtp(), [1, 2, 3], 4, k=8)
    assert m.calls == []
    with pytest.raises(RuntimeError, match="first verify"):  # k = 7 passes the check
        g.speculate(m, StubMtp(), [1, 2, 3], 4, k=7)
    assert m.verify_rows == [8]
    plain = Stub()  # without the kernels' head the limit is not applied
    with pytest.raises(RuntimeError, match="first verify"):
        g.speculate(plain, StubMtp(), [1, 2, 3], 4, k=8)
    assert plain.verify_rows == [9]


def test_head_rejects_more_than_eight_rows():
    seen = []
    head = kernels.Head.__new__(kernels.Head)
    head.head = lambda h, rows: seen.append(rows) or list(range(rows))
    assert head(None, 8) == list(range(8))
    with pytest.raises(ValueError, match="at most 8"):
        head(None, 9)
    assert seen == [8]


# L1


@pytest.mark.parametrize("fn", ["generate", "speculate"])
def test_zero_new_tokens_run_nothing(ttnn_stub, fn):
    m, seen = Stub(), []
    args = (m, [1, 2, 3], 0)
    out = (
        g.generate(*args, on_token=seen.append)
        if fn == "generate"
        else g.speculate(m, StubMtp(), *args[1:], on_token=seen.append)
    )
    assert out[0] == [] and m.calls == [] and seen == []


@pytest.mark.parametrize("fn", ["generate", "speculate"])
def test_negative_new_tokens_are_refused(ttnn_stub, fn):
    m = Stub()
    with pytest.raises(ValueError, match="-1"):
        g.generate(m, [1], -1) if fn == "generate" else g.speculate(m, StubMtp(), [1], -1)
    assert m.calls == []


# The command line, with the modules that need a device replaced


@pytest.fixture
def cli_env(monkeypatch, ttnn_stub):
    env = types.SimpleNamespace(loads=[], opened=[], closed=[], model=Stub(), want=None, tokens=[])

    def open_device(device_id=0, **kw):
        env.opened.append(device_id)
        return object()

    def load(ex, dev, **kw):
        env.loads.append(kw)
        return env.model

    class Export:
        def __init__(self, path):
            self.root, self.text_config = Path(path), {}

        def close(self):
            env.closed.append("export")

    for name, attrs in {
        "tensix_runner.device": {"open_device": open_device, "info": lambda d: "stub"},
        "tensix_runner.models": {"load": load},
        "tensix_runner.export": {"Export": Export},
    }.items():
        mod = types.ModuleType(name)
        mod.__dict__.update(attrs)
        monkeypatch.setitem(sys.modules, name, mod)
    monkeypatch.setattr(pcc, "prompt_tokens", lambda ex, n: env.tokens.append(n) or list(range(n)))
    return env


def run_cli(*argv):
    return cli.main(list(argv))


# L2


@pytest.mark.parametrize("argv", [["pcc", "exp"], ["pcc", "exp", "--decode", "0", "--window", "8"]])
def test_pcc_with_nothing_to_check_is_a_usage_error(cli_env, argv):
    with pytest.raises(SystemExit) as e:
        run_cli(*argv)
    assert e.value.code not in (0, None)
    assert "--layers" in str(e.value.code) and "--greedy" in str(e.value.code)
    assert cli_env.opened == [] and cli_env.loads == []


def test_mtp_check_gets_one_token_past_the_prompt(cli_env, monkeypatch):
    mtp = types.ModuleType("tensix_runner.models.qwen35")
    mtp.Mtp = lambda model: object()
    monkeypatch.setitem(sys.modules, "tensix_runner.models.qwen35", mtp)
    monkeypatch.setattr(pcc, "check_mtp", lambda *a, **kw: (1.0, 0.0, 1, [], []))
    assert run_cli("pcc", "exp", "--mtp", "--decode", "0", "--prefill", "32") == 0
    assert cli_env.tokens == [33]
    assert cli_env.loads[0]["max_seq"] >= 33


# M2 and M3


class FakeHF:
    """transformers' model on a ten-token vocabulary: step i's logits peak at 5, EOS, 5, 5, ...
    (the end-of-sequence token is the raw argmax at step 1)."""

    def __init__(self):
        self.steps = 0

    def __call__(self, input_ids, past_key_values=None, use_cache=False):
        row = torch.zeros(1, 1, 10)
        row[0, 0, [5, EOS][self.steps] if self.steps < 2 else 5] = 3.0
        self.steps += 1
        return types.SimpleNamespace(logits=row, past_key_values=object())


def test_reference_does_not_stop_early(monkeypatch):
    monkeypatch.setattr(pcc, "_hf_model", lambda export, overrides=None: FakeHF())
    export = types.SimpleNamespace(text_config={})
    want, logits = pcc.greedy_reference(export, [1, 2, 3], 8)
    assert len(want) == 8 and len(logits) == 8
    assert want[:3] == [5, EOS, 5]
    assert all(t == int(row.argmax()) for t, row in zip(want, logits))


def test_greedy_runs_every_step_past_an_end_of_sequence_token(cli_env, monkeypatch, capsys):
    monkeypatch.setattr(pcc, "_hf_model", lambda export, overrides=None: FakeHF())
    cli_env.model = Stub(eos_at=32)  # the device emits the end-of-sequence token at step 1 too
    assert run_cli("pcc", "exp", "--greedy", "8", "--prefill", "32") == 0
    decodes = [c for c in cli_env.model.calls if c[0] == "decode"]
    assert decodes == [("decode", 32 + k) for k in range(7)]  # 8 steps: 1 prefill, 7 decodes
    assert "greedy: 8 of 8 steps equal" in capsys.readouterr().out


def test_greedy_refuses_a_short_reference(cli_env, monkeypatch):
    monkeypatch.setattr(pcc, "greedy_reference", lambda *a, **kw: ([5] * 4, [torch.zeros(10)] * 4))
    with pytest.raises(ValueError, match="4 tokens"):
        run_cli("pcc", "exp", "--greedy", "8")


def test_compare_greedy_wants_equal_lengths():
    logits = [torch.zeros(10, dtype=torch.float64)] * 4
    for got, want, lg in [([1] * 3, [1] * 4, logits), ([1] * 4, [1] * 4, logits[:3])]:
        with pytest.raises(ValueError, match="same length"):
            pcc.compare_greedy(got, want, lg)
    assert pcc.compare_greedy([1] * 4, [1] * 4, logits)[0] == 4


def test_greedy_sizes_the_allocation(cli_env, monkeypatch):
    monkeypatch.setattr(
        pcc, "greedy_reference", lambda ex, p, n, o=None: ([5] * n, [torch.zeros(10)] * n)
    )
    run_cli("pcc", "exp", "--greedy", "1000", "--prefill", "32")
    assert cli_env.loads[0]["max_seq"] == 1056  # 32 + 1000, rounded up to 32
    cli_env.loads.clear()
    run_cli("pcc", "exp", "--greedy", "8")
    assert cli_env.loads[0]["max_seq"] == 256  # the minimum


def test_greedy_must_not_be_negative(cli_env):
    with pytest.raises(SystemExit, match="--greedy is -1"):
        run_cli("pcc", "exp", "--greedy", "-1")
    assert cli_env.opened == []


def test_spec_k_is_checked_before_the_device_opens(cli_env):
    with pytest.raises(SystemExit, match="--k"):
        run_cli("pcc", "exp", "--spec", "4", "--k", "0")
    assert cli_env.opened == []


# M4


class FakeTok:
    def encode(self, text, add_special_tokens=False):
        return types.SimpleNamespace(ids=list(range(100)))


@pytest.fixture
def bench_env(cli_env, monkeypatch):
    monkeypatch.setattr(g, "tokenizer", lambda ex: FakeTok())
    return cli_env


@pytest.mark.parametrize(
    "argv, words",
    [
        (["--pp", "512"], ("512", "100")),  # more than the prompt ids
        (["--pp", "64,200"], ("200", "100")),
        (["--pp", "64", "--depth", "150"], ("150", "100")),
        (["--pp", "64", "--max-seq", "32"], ("64", "32")),
        (["--pp", "32", "--depth", "90", "--tg", "20", "--max-seq", "100"], ("90", "100")),
        (["--pp", "32", "--depth", "0", "--tg", "200", "--max-seq", "100"], ("0", "100")),
    ],
)
def test_bench_lengths_are_checked_before_timing(bench_env, capsys, argv, words):
    with pytest.raises(SystemExit) as e:
        run_cli("bench", "exp", *argv)
    assert all(w in str(e.value.code) for w in words)
    assert bench_env.model.calls == [] and capsys.readouterr().out == ""


# L1 on the command line


def _run_env(cli_env, monkeypatch):
    cli_env.model = Stub(max_seq=1000)
    monkeypatch.setattr(g, "tokenizer", lambda ex: FakeTok())
    monkeypatch.setattr(g, "stop_ids", lambda ex: set())
    monkeypatch.setattr(g, "chat_prompt", lambda ex, msgs, thinking=False: "hi")


def test_run_with_zero_max_new_prefills_nothing(cli_env, monkeypatch, capsys):
    _run_env(cli_env, monkeypatch)
    assert run_cli("run", "exp", "hi", "--max-new", "0") == 0
    assert cli_env.model.calls == []
    assert "0 new" in capsys.readouterr().out


def test_run_with_negative_max_new_is_refused(cli_env, monkeypatch):
    _run_env(cli_env, monkeypatch)
    with pytest.raises(ValueError, match="-1"):
        run_cli("run", "exp", "hi", "--max-new", "-1")
    assert cli_env.model.calls == []


@pytest.mark.parametrize("fn", ["generate", "speculate"])
def test_empty_prompt_is_refused(fn):
    m = Stub()
    with pytest.raises(ValueError, match="prompt"):
        g.generate(m, [], 1) if fn == "generate" else g.speculate(m, StubMtp(), [], 1)
    assert m.calls == []


def test_generate_allows_exact_cache_capacity():
    m = Stub(max_seq=4)
    assert len(g.generate(m, [1, 2, 3, 4], 1)[0]) == 1
    assert len(g.generate(m, [1, 2, 3], 2)[0]) == 2
    with pytest.raises(ValueError, match="max_seq"):
        g.generate(m, [1, 2, 3], 3)


@pytest.mark.parametrize(
    "option,value", [("--pp", "0"), ("--depth", "-1"), ("--tg", "0"), ("--repeat", "0")]
)
def test_bench_rejects_invalid_counts(bench_env, option, value):
    with pytest.raises(SystemExit, match=option):
        run_cli("bench", "exp", option, value)
    assert bench_env.opened == []


@pytest.mark.parametrize(
    "args",
    [
        ["--verify", "1"],
        ["--decode", "-1"],
        ["--spec", "-1"],
        ["--window", "-1"],
        ["--layers", "0", "--prefill", "31"],
        ["--prefill", "0"],
    ],
)
def test_pcc_rejects_invalid_counts(cli_env, args):
    with pytest.raises(SystemExit):
        run_cli("pcc", "exp", "--greedy", "1", *args)
    assert cli_env.opened == []


def test_layer_selection_rejects_invalid_and_missing_layers():
    ex = types.SimpleNamespace(text_config={"num_hidden_layers": 3}, formats={(0, "p"): "bf16"})
    cli._check_layers(ex, {0})
    for layer in (-1, 1, 3):
        with pytest.raises(SystemExit, match="layer"):
            cli._check_layers(ex, {layer})


def test_load_failure_closes_export_even_if_device_close_fails(cli_env, monkeypatch, ttnn_stub):
    def fail_load(*args, **kwargs):
        raise ValueError("load failed")

    def fail_close(dev):
        raise RuntimeError("device close failed")

    monkeypatch.setattr(sys.modules["tensix_runner.models"], "load", fail_load)
    monkeypatch.setattr(ttnn_stub, "close_device", fail_close)
    with pytest.raises(RuntimeError, match="device close failed"):
        run_cli("run", "exp", "hi")
    assert cli_env.closed == ["export"]


def test_run_closes_export_even_if_device_close_fails(cli_env, monkeypatch, ttnn_stub):
    _run_env(cli_env, monkeypatch)

    def fail_close(dev):
        raise RuntimeError("close")

    monkeypatch.setattr(ttnn_stub, "close_device", fail_close)
    with pytest.raises(RuntimeError, match="close"):
        run_cli("run", "exp", "hi", "--max-new", "0")
    assert cli_env.closed == ["export"]


def test_load_failure_and_success_close_resources(cli_env, monkeypatch, ttnn_stub):
    devices = []
    monkeypatch.setattr(ttnn_stub, "close_device", devices.append)
    models = sys.modules["tensix_runner.models"]
    load = models.load

    def fail_load(*args, **kwargs):
        raise ValueError("load failed")

    monkeypatch.setattr(models, "load", fail_load)
    with pytest.raises(ValueError, match="load failed"):
        run_cli("run", "exp", "hi")
    monkeypatch.setattr(models, "load", load)
    _run_env(cli_env, monkeypatch)
    run_cli("run", "exp", "hi", "--max-new", "0")
    assert len(devices) == 2 and cli_env.closed == ["export", "export"]


@pytest.mark.parametrize("command,extra", [("check", []), ("pcc", ["--greedy", "1"])])
def test_empty_layer_selection_is_rejected_before_device(cli_env, command, extra):
    with pytest.raises(SystemExit, match="--layers requires"):
        run_cli(command, "exp", "--layers", ",", *extra)
    assert cli_env.opened == []


def test_empty_layer_set_is_rejected():
    with pytest.raises(SystemExit, match="--layers requires"):
        cli._check_layers(None, set())
