# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Xylolabs
"""Shared fixtures. `device` opens one chip (a card or ttsim) for the module and skips the tests that
take it when ttnn is missing or sees no chip; `tiny_export` writes a small synthetic export."""

import json
import os
from pathlib import Path

import pytest
import torch


def _ttnn_or_skip():
    try:
        import ttnn
    except ImportError:
        pytest.skip("ttnn is not installed")
    return ttnn


@pytest.fixture(scope="module")
def device():
    ttnn = _ttnn_or_skip()
    from tensix_runner.device import open_device

    try:
        dev = open_device()
    except RuntimeError as e:  # ttnn's TT_FATAL without a card or the simulator
        pytest.skip(f"ttnn opens no device: {str(e).splitlines()[0] if str(e) else e}")
    yield dev
    ttnn.close_device(dev)


@pytest.fixture
def export_dir():
    d = os.environ.get("TENSIX_RUNNER_EXPORT")
    if not d:
        pytest.skip("TENSIX_RUNNER_EXPORT names no export directory")
    return Path(d)


def write_export(root, tensors, layers, config, lm_head="bf16", index=True):
    """A synthetic export: tensors {name: tensor} in two shards (or one model.safetensors),
    quant4.json with layers {layer: {proj: format or {"format", "online"}}}, and config.json."""
    from safetensors.torch import save_file

    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    (root / "config.json").write_text(json.dumps(config))
    (root / "quant4.json").write_text(json.dumps({"layers": layers, "lm_head": lm_head}))
    names = sorted(tensors)
    if not index:
        save_file({n: tensors[n] for n in names}, root / "model.safetensors")
        return root
    half = len(names) // 2 or 1
    shards = {"a.safetensors": names[:half], "b.safetensors": names[half:]}
    weight_map = {}
    for shard, ns in shards.items():
        if ns:
            save_file({n: tensors[n] for n in ns}, root / shard)
            weight_map.update(dict.fromkeys(ns, shard))
    (root / "model.safetensors.index.json").write_text(json.dumps({"weight_map": weight_map}))
    return root


@pytest.fixture
def tiny_export(tmp_path):
    """A two-layer Qwen3-like export with bfp4 and bfp8 projections already on their grids."""
    from tensix_runner.bfp import bfp

    g = torch.Generator().manual_seed(0)

    def on(fmt, out_dim, in_dim):
        w = torch.randn(out_dim, in_dim, generator=g)
        bits = {"bfp4": 3, "bfp8": 7}[fmt]
        return bfp(w.T.contiguous(), bits).T.contiguous().to(torch.bfloat16)

    tensors, layers = {}, {}
    for i in range(2):
        layers[str(i)] = {"self_attn.q_proj": "bfp8", "mlp.down_proj": {"format": "bfp4"}}
        tensors[f"model.layers.{i}.self_attn.q_proj.weight"] = on("bfp8", 64, 32)
        tensors[f"model.layers.{i}.mlp.down_proj.weight"] = on("bfp4", 32, 64)
    tensors["model.embed_tokens.weight"] = torch.randn(64, 32, generator=g).to(torch.bfloat16)
    cfg = {"model_type": "qwen3", "num_hidden_layers": 2, "hidden_size": 32}
    return write_export(tmp_path / "tiny", tensors, layers, cfg)
