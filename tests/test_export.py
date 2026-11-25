# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Xylolabs
"""Export loading on synthetic exports: formats, prefixes, the head's name and tie rules, refusals,
and the host check."""

import json
from pathlib import Path

import pytest
import torch
from conftest import write_export

from tensix_runner.cli import main
from tensix_runner.export import Export, ExportError


def test_formats_and_names(tiny_export):
    ex = Export(tiny_export)
    assert ex.formats == {
        (0, "self_attn.q_proj"): "bfp8",
        (0, "mlp.down_proj"): "bfp4",
        (1, "self_attn.q_proj"): "bfp8",
        (1, "mlp.down_proj"): "bfp4",
    }
    assert ex.prefix == "model."
    assert ex.head_name == "model.embed_tokens.weight"  # tied: no lm_head.weight
    assert ex.head_format == "bf16"
    assert ex.model_type == "qwen3"
    assert [m[:3] for m in ex.matrices({1})] == [
        (1, "mlp.down_proj", "bfp4"),
        (1, "self_attn.q_proj", "bfp8"),
    ]
    w = ex.get(ex.name(0, "mlp.down_proj"))
    assert w.dtype == torch.bfloat16 and tuple(w.shape) == (32, 64)
    assert len(ex.sha1) == 40


def test_multimodal_prefix_and_head(tmp_path):
    t = {
        "model.language_model.layers.0.mlp.up_proj.weight": torch.zeros(
            32, 32, dtype=torch.bfloat16
        ),
        "model.visual.blocks.0.mlp.up_proj.weight": torch.zeros(32, 32, dtype=torch.bfloat16),
        "mtp.layers.0.mlp.up_proj.weight": torch.zeros(32, 32, dtype=torch.bfloat16),
        "lm_head.weight": torch.zeros(64, 32, dtype=torch.bfloat16),
    }
    cfg = {"model_type": "qwen3_5", "text_config": {"model_type": "qwen3_5_text"}}
    d = write_export(tmp_path / "mm", t, {"0": {"mlp.up_proj": "bfp4"}}, cfg, lm_head="bfp8")
    ex = Export(d)
    assert ex.prefix == "model.language_model."
    assert ex.model_type == "qwen3_5_text"
    assert ex.head_name == "lm_head.weight" and ex.head_format == "bfp8"


def test_one_file_checkpoint(tmp_path):
    t = {"model.layers.0.mlp.up_proj.weight": torch.zeros(32, 32, dtype=torch.bfloat16)}
    d = write_export(tmp_path / "one", t, {"0": {"mlp.up_proj": "bfp4"}}, {}, index=False)
    assert Export(d).weight_map == {"model.layers.0.mlp.up_proj.weight": "model.safetensors"}


@pytest.mark.parametrize(
    "layers, why",
    [
        ({"0": {"mlp.up_proj": "int4"}}, "not one of"),
        ({"0": {"mlp.up_proj": {"format": "bfp4", "online": True}}}, "online rotation"),
        ({}, "lists no layers"),
    ],
)
def test_refusals(tmp_path, layers, why):
    t = {"model.layers.0.mlp.up_proj.weight": torch.zeros(32, 32, dtype=torch.bfloat16)}
    d = write_export(tmp_path / "bad", t, layers, {})
    with pytest.raises(ExportError, match=why):
        Export(d)


def test_not_an_export(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps({}))
    with pytest.raises(ExportError, match="quant4.json"):
        Export(tmp_path)


def test_host_check(tiny_export, capsys):
    assert main(["check", str(tiny_export), "--host"]) == 0
    assert "5 of 5 matrices exact" in capsys.readouterr().out  # the bf16 head counts with --host
    # a value moved off the grid is reported and fails the check
    ex = Export(tiny_export)
    name = ex.name(1, "mlp.down_proj")
    shard = ex.weight_map[name]
    from safetensors.torch import load_file, save_file

    tensors = load_file(tiny_export / shard)
    g = torch.Generator().manual_seed(3)
    tensors[name] = torch.randn(tensors[name].shape, generator=g).to(torch.bfloat16)
    save_file(tensors, tiny_export / shard)
    assert main(["check", str(tiny_export), "--host"]) == 1
    assert "MOVED layer 1 mlp.down_proj (bfp4)" in capsys.readouterr().out


def test_unknown_family(tiny_export):
    from tensix_runner.models import load

    ex = Export(tiny_export)
    ex.config["model_type"] = "mamba"
    with pytest.raises(NotImplementedError, match="mamba"):
        load(ex, device=None)


def head_export(tmp_path, head=None, tied=None, nested=None, field="absent", embed=None):
    """An export with one layer, an embedding and optionally a head tensor named `head`;
    `lm_head_tensor` is set to `field` unless "absent"; `tied` and `nested` set
    tie_word_embeddings at the top level and in text_config."""
    g = torch.Generator().manual_seed(5)
    prefix = "model.language_model." if embed else "model."
    t = {
        f"{prefix}layers.0.mlp.up_proj.weight": torch.zeros(32, 32, dtype=torch.bfloat16),
        embed or "model.embed_tokens.weight": torch.randn(64, 32, generator=g).bfloat16(),
    }
    if head:
        t[head] = torch.randn(64, 32, generator=g).bfloat16()
    cfg = {"model_type": "qwen3"}
    if tied is not None:
        cfg["tie_word_embeddings"] = tied
    if nested is not None:
        cfg["text_config"] = {"model_type": "qwen3_text", "tie_word_embeddings": nested}
    extra = {} if field == "absent" else {"lm_head_tensor": field}
    d = write_export(tmp_path / "h", t, {"0": {"mlp.up_proj": "bfp4"}}, cfg, **extra)
    return Export(d)


def test_declared_head_name(tmp_path):
    name = "language_model.lm_head.weight"
    ex = head_export(tmp_path, head=name, tied=False, field=name)
    assert ex.head_name == name
    assert not torch.equal(ex.get(ex.head_name), ex.get(ex.embed_name))
    assert torch.equal(ex.get(ex.head_name), ex.get(name))


@pytest.mark.parametrize(
    "field", ["language_model.lm_head.weight", "lm_head.weight", 7, True, [], {}]
)
def test_declared_head_missing_or_not_a_name(tmp_path, field):
    ex = head_export(tmp_path, tied=True, field=field)
    with pytest.raises(ExportError, match="lm_head_tensor"):
        _ = ex.head_name


def test_null_field_counts_as_absent(tmp_path):
    assert head_export(tmp_path / "a", head="lm_head.weight", tied=False, field=None).head_name == (
        "lm_head.weight"
    )
    assert (
        head_export(tmp_path / "b", tied=True, field=None).head_name == "model.embed_tokens.weight"
    )
    with pytest.raises(ExportError, match="set lm_head_tensor"):
        _ = head_export(tmp_path / "c", tied=False, field=None).head_name


def test_declared_missing_embedding_falls_through(tmp_path):
    # a tied export names the embedding without the prefix the checkpoint stores
    ex = head_export(
        tmp_path,
        tied=True,
        field="model.embed_tokens.weight",
        embed="model.language_model.embed_tokens.weight",
    )
    assert ex.head_name == "model.language_model.embed_tokens.weight"


@pytest.mark.parametrize("field", ["absent", "lm_head.weight"])
def test_lm_head_weight_by_name(tmp_path, field):
    ex = head_export(tmp_path, head="lm_head.weight", tied=False, field=field)
    assert ex.head_name == "lm_head.weight"


def test_no_head_tensor_uses_the_embedding_when_tied(tmp_path):
    assert head_export(tmp_path, tied=True).head_name == "model.embed_tokens.weight"


def test_untied_head_under_another_name_fails(tmp_path):
    ex = head_export(tmp_path, head="language_model.lm_head.weight", tied=False)
    with pytest.raises(ExportError, match="lm_head_tensor"):
        _ = ex.head_name


@pytest.mark.parametrize(
    "tied, nested, ok",
    [
        (True, None, True),
        (None, True, True),
        (True, False, True),  # the top level wins
        (None, False, False),
        (False, True, False),
        (None, None, False),
    ],
)
def test_tie_word_embeddings_levels(tmp_path, tied, nested, ok):
    ex = head_export(tmp_path, tied=tied, nested=nested)
    if ok:
        assert ex.head_name == ex.embed_name
    else:
        with pytest.raises(ExportError, match="set lm_head_tensor"):
            _ = ex.head_name


def on_grid_weight(fmt, shape, seed):
    from tensix_runner.bfp import bfp

    w = torch.randn(shape, generator=torch.Generator().manual_seed(seed))
    return bfp(w.T.contiguous(), {"bfp4": 3, "bfp8": 7}[fmt]).T.contiguous().bfloat16()


def check_export(tmp_path, head_fmt="bfp4", head_on_grid=True, tied=False):
    """Two layers on their grids and a head: lm_head.weight, or the embedding where tied."""
    t = {
        f"model.layers.{i}.mlp.up_proj.weight": on_grid_weight("bfp4", (32, 32), i) for i in (0, 1)
    }
    head = (
        on_grid_weight(head_fmt, (64, 32), 7)
        if head_fmt != "bf16"
        else torch.randn(64, 32).bfloat16()
    )
    if not head_on_grid:
        head = torch.randn(64, 32, generator=torch.Generator().manual_seed(8)).bfloat16()
    t["model.embed_tokens.weight" if tied else "lm_head.weight"] = head
    if not tied:
        t["model.embed_tokens.weight"] = torch.randn(64, 32).bfloat16()
    layers = {"0": {"mlp.up_proj": "bfp4"}, "1": {"mlp.up_proj": "bfp4"}}
    cfg = {"model_type": "qwen3", "tie_word_embeddings": tied}
    return str(write_export(tmp_path / "c", t, layers, cfg, lm_head=head_fmt))


def test_check_flags_a_head_off_the_grid(tmp_path, capsys):
    d = check_export(tmp_path, head_on_grid=False)
    assert main(["check", d, "--host"]) == 1
    out = capsys.readouterr().out
    assert "MOVED head (bfp4)" in out and "MOVED layer" not in out
    assert "2 of 3 matrices exact" in out


@pytest.mark.parametrize("fmt", ["bfp4", "bfp8"])
def test_check_counts_a_head_on_the_grid(tmp_path, capsys, fmt):
    assert main(["check", check_export(tmp_path, head_fmt=fmt), "--host"]) == 0
    assert "3 of 3 matrices exact" in capsys.readouterr().out  # two layers and the head


def test_check_skips_a_bf16_head_without_host(tmp_path, capsys, monkeypatch):
    from tensix_runner import bfp, weights

    monkeypatch.setattr(weights, "check", lambda w, fmt, dev=None: bfp.on_grid(w, fmt))
    d = check_export(tmp_path, head_fmt="bf16")
    assert main(["check", d]) == 0
    assert "2 of 2 matrices exact" in capsys.readouterr().out
    assert main(["check", d, "--host"]) == 0
    assert "3 of 3 matrices exact" in capsys.readouterr().out


def test_check_with_layers_still_checks_the_head(tmp_path, capsys):
    d = check_export(tmp_path, head_on_grid=False)
    assert main(["check", d, "--host", "--layers", "1"]) == 1
    out = capsys.readouterr().out
    assert "MOVED head (bfp4)" in out and "1 of 2 matrices exact" in out
    assert main(["check", check_export(tmp_path / "ok"), "--host", "--layers", "0"]) == 0
    assert "2 of 2 matrices exact" in capsys.readouterr().out


def test_check_reads_a_tied_head_as_the_embedding(tmp_path, capsys):
    assert main(["check", check_export(tmp_path / "a", tied=True), "--host"]) == 0
    assert "3 of 3 matrices exact" in capsys.readouterr().out
    d = check_export(tmp_path / "b", tied=True, head_on_grid=False)
    assert main(["check", d, "--host"]) == 1
    assert "MOVED head (bfp4)" in capsys.readouterr().out


def test_check_stops_at_a_missing_head(tmp_path):
    d = check_export(tmp_path)
    (Path(d) / "quant4.json").write_text(
        json.dumps({"layers": {"0": {"mlp.up_proj": "bfp4"}}, "lm_head_tensor": "x.lm_head.weight"})
    )
    with pytest.raises(ExportError, match="lm_head_tensor"):
        main(["check", d, "--host"])
