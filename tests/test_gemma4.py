# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Xylolabs
"""Gemma 4 on the host: which layer's KV cache each layer reads, and the RoPE frequencies of both
layer types (against transformers where it is installed). The device test runs a KV-sharing layer
and the layer it reads on a Gemma 4 export (TENSIX_RUNNER_EXPORT), each at PCC 0.99 or better."""

import pytest
import torch

from tensix_runner.models.gemma4 import FULL, SLIDING, Config

# gemma-4-E2B's text config, the values the runner reads
E2B = {
    "hidden_size": 1536,
    "num_hidden_layers": 35,
    "layer_types": [FULL if i % 5 == 4 else SLIDING for i in range(35)],
    "num_attention_heads": 8,
    "num_key_value_heads": 1,
    "head_dim": 256,
    "global_head_dim": 512,
    "rms_norm_eps": 1e-6,
    "vocab_size": 262144,
    "sliding_window": 512,
    "num_kv_shared_layers": 20,
    "hidden_size_per_layer_input": 256,
    "final_logit_softcapping": 30.0,
    "hidden_activation": "gelu_pytorch_tanh",
    "attention_k_eq_v": False,
    "enable_moe_block": False,
    "rope_parameters": {
        FULL: {"partial_rotary_factor": 0.25, "rope_theta": 1e6, "rope_type": "proportional"},
        SLIDING: {"rope_theta": 1e4, "rope_type": "default"},
    },
}


def test_kv_sources():
    c = Config(E2B)
    assert c.first_shared == 15
    assert [c.source(i) for i in range(15)] == list(range(15))
    # the last sliding layer before 15 is 13, the last full one 14
    assert all(c.source(i) == (14 if i % 5 == 4 else 13) for i in range(15, 35))


def test_rope_frequencies():
    c = Config(E2B)
    f = c.inv_freq(FULL)
    assert f.shape == (256,)
    want = 1.0 / (1e6 ** (torch.arange(0, 128, 2).float() / 512))
    assert torch.equal(f[:64], want) and torch.equal(f[64:], torch.zeros(192))
    s = c.inv_freq(SLIDING)
    assert torch.allclose(s, 1.0 / (1e4 ** (torch.arange(0, 256, 2).float() / 256)))


def test_rope_frequencies_match_transformers():
    pytest.importorskip("transformers")
    from transformers import Gemma4TextConfig
    from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS

    hf = Gemma4TextConfig(**E2B)
    c = Config(E2B)
    got, _ = ROPE_INIT_FUNCTIONS["proportional"](hf, "cpu", layer_type=FULL)
    assert torch.allclose(c.inv_freq(FULL), got.float())


def test_unsupported_variants():
    for key, value in (("attention_k_eq_v", True), ("enable_moe_block", True)):
        with pytest.raises(NotImplementedError):
            Config({**E2B, key: value})
    with pytest.raises(NotImplementedError):
        Config({**E2B, "hidden_size_per_layer_input": 0})


@pytest.mark.device
@pytest.mark.export
def test_shared_layer(export_dir, device):
    pytest.importorskip("transformers")
    from tensix_runner import pcc
    from tensix_runner.export import Export
    from tensix_runner.models import load

    ex = Export(export_dir)
    if not ex.model_type.startswith("gemma4"):
        pytest.skip(f"{ex.model_type} is not a Gemma 4 export")
    c = Config(ex.text_config)
    i = c.first_shared
    model = load(ex, device, max_seq=256, layers=[i], head=False)
    res = pcc.check_layers(model, pcc.prompt_tokens(ex, 33), 32, [i], log=lambda s: None)
    assert {k[0] for k in res} == {c.source(i), i}
    assert all(r >= 0.99 for r, _ in res.values()), res
