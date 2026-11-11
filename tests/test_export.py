# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Xylolabs
"""Export loading on synthetic exports: formats, prefixes, the tied head, refusals, and the host check."""

import json

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
    assert "4 of 4 matrices exact" in capsys.readouterr().out
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
