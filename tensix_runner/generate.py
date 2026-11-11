# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Xylolabs
"""Prompts, stop tokens and the greedy loop.

The chat template is the checkpoint's own (chat_template.jinja, or tokenizer_config.json's
chat_template), rendered with Jinja2 as transformers renders it. Decoding is greedy: the argmax runs
on the device and only the chosen token id comes back."""

import json
import time
from pathlib import Path


def tokenizer(export):
    from tokenizers import Tokenizer

    return Tokenizer.from_file(str(export.root / "tokenizer.json"))


def chat_template(export):
    root = Path(export.root)
    f = root / "chat_template.jinja"
    if f.exists():
        return f.read_text()
    cfg = json.loads((root / "tokenizer_config.json").read_text())
    t = cfg.get("chat_template")
    if not t:
        raise ValueError(f"{root}: no chat template")
    return t


def chat_prompt(export, messages, thinking=False):
    """The text the chat template makes of messages, ending with the assistant's turn."""
    from jinja2.sandbox import ImmutableSandboxedEnvironment

    def raise_exception(msg):
        raise ValueError(msg)

    env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True)
    env.filters["tojson"] = lambda x, indent=None, **kw: json.dumps(
        x, ensure_ascii=False, indent=indent
    )
    env.globals["raise_exception"] = raise_exception
    cfg = json.loads((Path(export.root) / "tokenizer_config.json").read_text())
    return env.from_string(chat_template(export)).render(
        messages=messages,
        add_generation_prompt=True,
        enable_thinking=thinking,
        bos_token=cfg.get("bos_token") or "",
        eos_token=cfg.get("eos_token") or "",
    )


def stop_ids(export):
    """generation_config.json's eos_token_id (one id or a list)."""
    f = Path(export.root) / "generation_config.json"
    ids = json.loads(f.read_text()).get("eos_token_id") if f.exists() else None
    if ids is None:
        return set()
    return set(ids) if isinstance(ids, list) else {ids}


def generate(model, tokens, max_new, stop=(), on_token=None):
    """Greedy tokens after the prompt tokens, until a stop token or max_new. Returns (tokens,
    prefill seconds, decode seconds)."""
    if len(tokens) + max_new > model.max_seq:
        raise ValueError(
            f"{len(tokens)} prompt + {max_new} new tokens exceed max_seq {model.max_seq}"
        )
    t0 = time.perf_counter()
    out = [model.prefill(tokens)]
    t1 = time.perf_counter()
    if on_token:
        on_token(out[-1])
    pos = len(tokens)
    while len(out) < max_new and out[-1] not in stop:
        out.append(model.decode(out[-1], pos))
        pos += 1
        if on_token:
            on_token(out[-1])
    return out, t1 - t0, time.perf_counter() - t1


def speculate(model, mtp, tokens, max_new, k=2, stop=(), on_token=None):
    """Greedy tokens with k drafts a round from the model's MTP module (mtp.draft), each round
    verified as one decode step of k + 1 rows (model.verify) and the kept rows committed: the
    output equals generate()'s up to ties within bf16 rounding. Returns (tokens, prefill seconds,
    decode seconds, the number of draft rows each round kept)."""
    import ttnn

    if len(tokens) + max_new + k > model.max_seq:
        raise ValueError(
            f"{len(tokens)} prompt + {max_new} new tokens exceed max_seq {model.max_seq}"
        )
    t0 = time.perf_counter()
    first, x = model.prefill_hidden(tokens)
    drafts = mtp.draft(x, list(tokens[1:]) + [first], 0, k)
    ttnn.deallocate(x)
    t1 = time.perf_counter()
    out, kept, p = [first], [], len(tokens)
    if on_token:
        on_token(first)
    while len(out) < max_new and out[-1] not in stop:
        ver, xv = model.verify([out[-1], *drafts], p)
        m = 0
        while m < k and drafts[m] == ver[m]:
            m += 1
        model.commit(m + 1)
        new = ver[: m + 1]
        kept.append(m)
        for t in new:
            if len(out) < max_new and out[-1] not in stop:
                out.append(t)
                if on_token:
                    on_token(t)
        if len(out) >= max_new or out[-1] in stop:  # no drafts needed after the last round
            ttnn.deallocate(xv)
            break
        rows = ttnn.slice(xv, [0, 0, 0, 0], [1, 1, m + 1, xv.shape[-1]])
        drafts = mtp.draft(rows, new, p, k)
        ttnn.deallocate(xv)
        p += m + 1
    return out, t1 - t0, time.perf_counter() - t1, kept
