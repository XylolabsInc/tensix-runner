# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Xylolabs
"""The greedy check's bookkeeping on the host: bfloat16's spacing and the tie rule."""

import torch

from tensix_runner import pcc


def test_bf16_ulp():
    assert pcc.bf16_ulp(15.6) == 0.0625
    assert pcc.bf16_ulp(-1.0) == 2**-7
    assert pcc.bf16_ulp(100.0) == 0.5


def test_compare_greedy():
    logits = [torch.zeros(10, dtype=torch.float64) for _ in range(3)]
    logits[1][4], logits[1][5] = 15.634, 15.585  # 0.049 apart: within one step (0.0625)
    logits[2][6], logits[2][7] = 15.6, 15.0
    same, ties, wrong = pcc.compare_greedy([1, 5, 7], [1, 4, 6], logits)
    assert same == 1
    assert [t[:3] for t in ties] == [(1, 5, 4)]
    assert [w[:3] for w in wrong] == [(2, 7, 6)]


def test_compare_greedy_rejects_a_device_token_above_the_reference():
    logits = [torch.zeros(10, dtype=torch.float64) for _ in range(3)]
    logits[0][4], logits[0][5] = 1.0, 1.01  # the device's token (5) is above the reference's (4)
    logits[1][4], logits[1][5] = 1.0, float("nan")
    logits[2][4], logits[2][5] = float("-inf"), float("-inf")  # the gap is nan
    same, ties, wrong = pcc.compare_greedy([5, 5, 5], [4, 4, 4], logits)
    assert same == 0 and ties == []
    assert [w[:3] for w in wrong] == [(0, 5, 4), (1, 5, 4), (2, 5, 4)]


def test_compare_greedy_counts_an_equal_pair_as_a_tie():
    logits = [torch.zeros(10, dtype=torch.float64)]
    assert pcc.compare_greedy([5], [4], logits)[1][0][:3] == (0, 5, 4)


def test_greedy_reference_keeps_an_end_of_sequence_argmax():
    """With the end-of-sequence token as the raw argmax, the reference token is that argmax."""
    import types

    import pytest

    transformers = pytest.importorskip("transformers")
    cfg = transformers.Qwen3Config(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=16,
    )
    torch.manual_seed(0)
    model = transformers.Qwen3ForCausalLM(cfg).eval()
    tokens = [1, 2, 3, 4]
    with torch.no_grad():
        first = int(model(input_ids=torch.tensor([tokens])).logits[0, -1].argmax())
    model.generation_config.eos_token_id = first
    orig = pcc._hf_model
    pcc._hf_model = lambda export, overrides=None: model
    try:
        want, logits = pcc.greedy_reference(types.SimpleNamespace(text_config={}), tokens, 4)
    finally:
        pcc._hf_model = orig
    assert want[0] == first and len(want) == len(logits) == 4
    assert all(t == int(row.argmax()) for t, row in zip(want, logits))


def test_prompt_tokens_have_one_bos_with_a_bos_post_processor(tmp_path):
    """A tokenizer whose post-processor adds BOS does not add a second one after template_bos."""
    import json
    import types

    from tokenizers import Tokenizer, models, pre_tokenizers, processors

    words = ["<bos>", "[UNK]", "The", "Tensix", "core", "pairs", "five", "small"]
    tok = Tokenizer(models.WordLevel({w: i for i, w in enumerate(words)}, unk_token="[UNK]"))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    tok.post_processor = processors.TemplateProcessing(
        single="<bos> $A", special_tokens=[("<bos>", 0)]
    )
    tok.save(str(tmp_path / "tokenizer.json"))
    (tmp_path / "tokenizer_config.json").write_text(
        json.dumps({"bos_token": "<bos>", "chat_template": "<bos>{{ messages[0].content }}"})
    )
    ids = pcc.prompt_tokens(types.SimpleNamespace(root=tmp_path), 4)
    assert ids.count(0) == 1 and ids[0] == 0
