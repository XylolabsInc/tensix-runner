# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Xylolabs
"""Layer and model checks against transformers on the same export (the `ref` extra).

The reference runs the export in float32 on the CPU and records every decoder layer's input and
output at every position. For each checked layer, the device layer gets the reference's input rows:
a prefill over the first T positions, then one decode step per later position on the KV rows the
prefill wrote. Each output is compared with the reference's by the Pearson correlation (PCC) and
the largest absolute difference. The whole-model check compares greedy tokens."""

import time

import torch


def pcc(a, b):
    a, b = a.double().flatten(), b.double().flatten()
    a, b = a - a.mean(), b - b.mean()
    return float((a @ b) / (a.norm() * b.norm()).clamp_min(1e-30))


def prompt_tokens(export, n):
    """The first n tokens of a fixed English text, from the export's tokenizer, after the BOS token
    where the chat template starts with one (Gemma)."""
    from . import generate

    tok = generate.tokenizer(export)
    text = (
        "The Tensix core pairs five small RISC-V processors with a matrix engine and a vector "
        "engine. Data moves between cores over two networks on chip, and each core keeps its "
        "working set in about one and a half megabytes of local SRAM. Large models stream their "
        "weights from DRAM for every token, so decode speed is set by memory bandwidth."
    )
    ids = generate.template_bos(export, tok) + tok.encode(text).ids
    assert len(ids) >= n, f"the text has {len(ids)} tokens, fewer than {n}"
    return ids[:n]


def _hf_model(export, overrides=None):
    """The export in transformers, float32 on the CPU; overrides: text config values to change."""
    from transformers import AutoConfig, AutoModelForCausalLM

    kw = {}
    if overrides:
        cfg = AutoConfig.from_pretrained(str(export.root))
        tc = getattr(cfg, "text_config", cfg)
        for k, v in overrides.items():
            setattr(tc, k, v)
        kw["config"] = cfg
    return AutoModelForCausalLM.from_pretrained(str(export.root), dtype=torch.float32, **kw)


def reference(export, tokens, layers, overrides=None):
    """{layer: (input [n, dim], output [n, dim], per-layer input [n, d] or None)} from
    transformers in float32 on the CPU. overrides: text config values to change (pcc's --window)."""
    m = _hf_model(export, overrides)
    m.eval()
    rec = {}
    # the decoder layers: model.layers (text models) or model.language_model.layers (multimodal)
    inner = m.model
    blocks = inner.layers if hasattr(inner, "layers") else inner.language_model.layers

    def hook(i):
        def f(mod, args, kwargs, out):
            x = kwargs.get("hidden_states", args[0] if args else None)
            # Gemma 4's layers take their per-layer input second
            ple = args[1] if len(args) > 1 and torch.is_tensor(args[1]) else None
            ple = kwargs.get("per_layer_input", ple)
            y = out[0] if isinstance(out, tuple) else out
            rec[i] = (
                x[0].detach().clone(),
                y[0].detach().clone(),
                None if ple is None else ple[0].detach().clone(),
            )

        return f

    hs = [blocks[i].register_forward_hook(hook(i), with_kwargs=True) for i in layers]
    with torch.no_grad():
        m(input_ids=torch.tensor([tokens]))
    for h in hs:
        h.remove()
    return rec


def greedy_reference(export, tokens, n, overrides=None):
    """transformers' greedy continuation (n tokens) in float32 on the CPU."""
    m = _hf_model(export, overrides)
    with torch.no_grad():
        out = m.generate(torch.tensor([tokens]), max_new_tokens=n, do_sample=False)
    return out[0, len(tokens) :].tolist()


def to_device(device, rows, logical=None):
    """Reference rows [n, dim] as a bf16 tile tensor [1, 1, n, dim] (logical rows padded to 32)."""
    import ttnn

    n, d = rows.shape
    pad = -(-n // 32) * 32
    t = torch.zeros(1, 1, pad, d)
    t[0, 0, :n] = rows
    x = ttnn.from_torch(
        t.to(torch.bfloat16), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
    )
    if logical is not None:
        x = ttnn.reshape(x, (1, 1, logical, d), (1, 1, pad, d))
    return x


def check_layers(model, tokens, prefill_n, layers, log=print, overrides=None):
    """Each layer on the reference's inputs: prefill of the first prefill_n positions (a multiple
    of 32), then decode of every later position. A layer that attends over an earlier layer's KV
    cache runs after that layer, which is checked too. Returns {(layer, "prefill" or position or
    "per-layer input"): (pcc, largest difference)}."""
    import ttnn

    run = sorted(set(layers) | {model.kv_source(i) for i in layers})
    ref = reference(model.export, tokens, run, overrides)
    out = {}

    def record(key, got, want, t0, what):
        out[key] = (pcc(got, want), float((got - want).abs().max()))
        log(
            f"layer {key[0]} {what}: pcc {out[key][0]:.6f}, max diff {out[key][1]:.4g} "
            f"({time.monotonic() - t0:.0f} s)"
        )

    for i in run:
        x, y, ple = ref[i]
        if ple is not None:
            t0 = time.monotonic()
            got = model.per_layer_input(i, tokens[:prefill_n])
            record((i, "per-layer input"), got, ple[:prefill_n], t0, "per-layer input")
        t0 = time.monotonic()
        got = model.check_prefill(i, to_device(model.device, x[:prefill_n]), tokens[:prefill_n])
        got = ttnn.to_torch(got)
        got = got.reshape(-1, got.shape[-1])[:prefill_n].float()
        record((i, "prefill"), got, y[:prefill_n], t0, f"prefill {prefill_n}")
        for p in range(prefill_n, len(tokens)):
            t0 = time.monotonic()
            xd = to_device(model.device, x[p : p + 1], logical=1)
            got = ttnn.to_torch(model.check_decode(i, xd, tokens[p], p))
            got = got.reshape(-1, got.shape[-1])[:1].float()
            record((i, p), got, y[p : p + 1], t0, f"decode at {p}")
    return out


def check_verify(model, tokens, n, T, keep=2, log=print):
    """A verify of T teacher-forced rows against T decode steps, on the same prompt: the greedy
    ids must agree row for row, and after commit(keep) the next decode step must give the
    sequential run's id at that position. Before the verify, a verify of other tokens fills the
    same KV rows, so a row the real verify fails to write shows. Returns (verify ids, sequential
    ids, id after commit)."""
    assert n + T <= len(tokens) and 0 < keep < T
    rows = tokens[n : n + T]
    t0 = time.monotonic()
    model.prefill(tokens[:n])
    model.verify([tokens[0]] * T, n)  # other K and V in rows n .. n + T - 1, not committed
    model.prefill(tokens[:n])
    ver, _ = model.verify(rows, n)
    model.commit(keep)
    after = model.decode(rows[keep], n + keep)
    log(f"verify: {ver}, after commit({keep}): {after} ({time.monotonic() - t0:.0f} s)")
    t0 = time.monotonic()
    model.prefill(tokens[:n])
    seq = [model.decode(t, n + j) for j, t in enumerate(rows)]
    log(f"sequential: {seq} ({time.monotonic() - t0:.0f} s)")
    return ver, seq, after
