# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Xylolabs
"""Prompts, stop tokens and the greedy loop.

The chat template is the checkpoint's own (chat_template.jinja, or tokenizer_config.json's
chat_template), rendered with Jinja2 as transformers renders it. Decoding is greedy: the argmax runs
on the device and only the chosen token id comes back."""

import json
import time
from pathlib import Path

from .kernels import HEAD_ROWS


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


def template_bos(export, tok):
    """[the BOS token's id] if the chat template starts its text with the BOS token, else []: a
    prompt given without the template starts the same way."""
    bos = json.loads((Path(export.root) / "tokenizer_config.json").read_text()).get("bos_token")
    if not bos or tok.token_to_id(bos) is None:
        return []
    try:
        text = chat_prompt(export, [{"role": "user", "content": "x"}])
    except ValueError:  # no chat template
        return []
    return [tok.token_to_id(bos)] if text.startswith(bos) else []


def stop_ids(export):
    """EOS ids from generation config, then the model config (one id or a list)."""
    f = Path(export.root) / "generation_config.json"
    ids = json.loads(f.read_text()).get("eos_token_id") if f.exists() else None
    if ids is None:
        ids = export.text_config.get("eos_token_id")
    if ids is None:
        ids = export.config.get("eos_token_id")
    if ids is None:
        return set()
    return set(ids) if isinstance(ids, list) else {ids}


def check_count(max_new):
    """max_new, a count of new tokens: an integer, not negative (ValueError)."""
    if isinstance(max_new, bool) or not isinstance(max_new, int) or max_new < 0:
        raise ValueError(
            f"max_new is {max_new!r}: the number of new tokens must be a non-negative integer"
        )


def check_k(k, kernels_head=False):
    """k, the drafts a round: an integer of at least 1 (ValueError); with the kernels' head, at
    most HEAD_ROWS - 1, as a round sends k + 1 rows to it."""
    if isinstance(k, bool) or not isinstance(k, int) or k < 1:
        raise ValueError(f"k is {k!r}: the drafts a round must be an integer of at least 1")
    if kernels_head and k + 1 > HEAD_ROWS:
        raise ValueError(
            f"k is {k}: a round sends {k + 1} rows, and the kernels head takes at most "
            f"{HEAD_ROWS} rows a call, so the largest k is {HEAD_ROWS - 1}"
        )


def generate(model, tokens, max_new, stop=(), on_token=None):
    """Greedy tokens after the prompt tokens, until a stop token or max_new. max_new 0 returns no
    tokens and runs nothing. Returns (tokens, prefill seconds, decode seconds)."""
    check_count(max_new)
    if max_new == 0:
        return [], 0.0, 0.0
    if not tokens:
        raise ValueError("the prompt must contain at least one token")
    if len(tokens) + max_new - 1 > model.max_seq:
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
    decode seconds, the number of draft rows each round kept). k must be an integer of at least 1,
    and at most 7 where the model has the kernels' head (k + 1 rows a call, 8 at most); max_new 0
    returns no tokens and runs nothing."""
    check_k(k, getattr(model, "khead", None) is not None)
    check_count(max_new)
    if max_new == 0:
        return [], 0.0, 0.0, []
    if not tokens:
        raise ValueError("the prompt must contain at least one token")
    import ttnn

    # one new token returns after the prefill: no draft rows are written
    if len(tokens) + (0 if max_new == 1 else max_new + k) > model.max_seq:
        raise ValueError(
            f"{len(tokens)} prompt + {max_new} new tokens exceed max_seq {model.max_seq}"
        )
    t0 = time.perf_counter()
    first, x = model.prefill_hidden(tokens)
    if max_new == 1 or first in stop:  # complete already: no drafts are needed
        ttnn.deallocate(x)
        t1 = time.perf_counter()
        if on_token:
            on_token(first)
        return [first], t1 - t0, 0.0, []
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
        a = 0  # the rows emitted: the accepted drafts and the next token, up to max_new or a stop
        for t in ver[: m + 1]:
            if len(out) >= max_new or out[-1] in stop:
                break
            out.append(t)
            a += 1
            if on_token:
                on_token(t)
        new = out[len(out) - a :]
        model.commit(a)  # only what was returned: the state never runs ahead of the output
        kept.append(a - 1)
        if len(out) >= max_new or out[-1] in stop:  # no drafts needed after the last round
            ttnn.deallocate(xv)
            break
        rows = ttnn.slice(xv, [0, 0, 0, 0], [1, 1, a, xv.shape[-1]])
        drafts = mtp.draft(rows, new, p, k)
        ttnn.deallocate(xv)
        p += a
    return out, t1 - t0, time.perf_counter() - t1, kept
