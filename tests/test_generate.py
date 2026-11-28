# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Xylolabs
"""The chat template, the stop tokens and the greedy loop's bookkeeping, on the host."""

import json

import pytest

from tensix_runner import generate as g
from tensix_runner.export import Export

TEMPLATE = (
    "{% for m in messages %}<|im_start|>{{ m.role }}\n{{ m.content }}<|im_end|>\n{% endfor %}"
    "{% if add_generation_prompt %}<|im_start|>assistant\n"
    "{% if not enable_thinking %}<think>\n\n</think>\n\n{% endif %}{% endif %}"
    "{{ {'n': messages | length} | tojson }}"
)


def test_chat_prompt(tiny_export):
    (tiny_export / "tokenizer_config.json").write_text(
        json.dumps({"chat_template": TEMPLATE, "eos_token": "<|im_end|>"})
    )
    ex = Export(tiny_export)
    text = g.chat_prompt(ex, [{"role": "user", "content": "hi"}])
    assert text == (
        '<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n{"n": 1}'
    )
    assert "<think>" not in g.chat_prompt(ex, [{"role": "user", "content": "hi"}], thinking=True)
    # a template file beside the config wins
    (tiny_export / "chat_template.jinja").write_text("{{ messages[0].content }}!")
    assert g.chat_prompt(ex, [{"role": "user", "content": "hi"}]) == "hi!"


class Tok:
    def token_to_id(self, t):
        return {"<bos>": 2}.get(t)


def test_template_bos(tiny_export):
    cfg = tiny_export / "tokenizer_config.json"
    cfg.write_text(
        json.dumps(
            {"chat_template": "{{ bos_token }}{{ messages[0].content }}", "bos_token": "<bos>"}
        )
    )
    assert g.template_bos(Export(tiny_export), Tok()) == [2]
    cfg.write_text(json.dumps({"chat_template": TEMPLATE, "bos_token": "<bos>"}))
    assert g.template_bos(Export(tiny_export), Tok()) == []
    cfg.write_text(json.dumps({"chat_template": TEMPLATE}))
    assert g.template_bos(Export(tiny_export), Tok()) == []


@pytest.mark.parametrize("ids, want", [([151645, 151643], {151645, 151643}), (2, {2})])
def test_stop_ids(tiny_export, ids, want):
    (tiny_export / "generation_config.json").write_text(json.dumps({"eos_token_id": ids}))
    assert g.stop_ids(Export(tiny_export)) == want


class Fake:
    """A model whose greedy token is always the position plus one."""

    max_seq = 64

    def prefill(self, tokens):
        return len(tokens)

    def decode(self, token, pos):
        return pos + 1


def test_generate_stops():
    out, _, _ = g.generate(Fake(), [7, 7, 7], max_new=10, stop={6})
    assert out == [3, 4, 5, 6]
    out, _, _ = g.generate(Fake(), [7], max_new=3)
    assert out == [1, 2, 3]
    with pytest.raises(ValueError, match="max_seq"):
        g.generate(Fake(), [7] * 60, max_new=10)


def test_stop_ids_fall_back_to_model_config(tmp_path):
    from types import SimpleNamespace

    ex = SimpleNamespace(
        root=tmp_path, text_config={"eos_token_id": [3, 4]}, config={"eos_token_id": 5}
    )
    assert g.stop_ids(ex) == {3, 4}
    (tmp_path / "generation_config.json").write_text('{"eos_token_id": null}')
    ex.text_config = {}
    assert g.stop_ids(ex) == {5}
    (tmp_path / "generation_config.json").write_text('{"eos_token_id": 6}')
    assert g.stop_ids(ex) == {6}
