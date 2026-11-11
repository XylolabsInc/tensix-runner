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
    """The first n tokens of a fixed English text, from the export's tokenizer."""
    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(str(export.root / "tokenizer.json"))
    text = (
        "The Tensix core pairs five small RISC-V processors with a matrix engine and a vector "
        "engine. Data moves between cores over two networks on chip, and each core keeps its "
        "working set in about one and a half megabytes of local SRAM. Large models stream their "
        "weights from DRAM for every token, so decode speed is set by memory bandwidth."
    )
    ids = tok.encode(text).ids
    assert len(ids) >= n, f"the text has {len(ids)} tokens, fewer than {n}"
    return ids[:n]


def reference(export, tokens, layers):
    """{layer: (input [n, dim], output [n, dim])} from transformers in float32 on the CPU."""
    from transformers import AutoModelForCausalLM

    m = AutoModelForCausalLM.from_pretrained(str(export.root), dtype=torch.float32)
    m.eval()
    rec = {}
    # the decoder layers: model.layers (text models) or model.language_model.layers (multimodal)
    inner = m.model
    blocks = inner.layers if hasattr(inner, "layers") else inner.language_model.layers

    def hook(i):
        def f(mod, args, kwargs, out):
            x = kwargs.get("hidden_states", args[0] if args else None)
            y = out[0] if isinstance(out, tuple) else out
            rec[i] = (x[0].detach().clone(), y[0].detach().clone())

        return f

    hs = [blocks[i].register_forward_hook(hook(i), with_kwargs=True) for i in layers]
    with torch.no_grad():
        m(input_ids=torch.tensor([tokens]))
    for h in hs:
        h.remove()
    return rec


def greedy_reference(export, tokens, n):
    """transformers' greedy continuation (n tokens) in float32 on the CPU."""
    from transformers import AutoModelForCausalLM

    m = AutoModelForCausalLM.from_pretrained(str(export.root), dtype=torch.float32)
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


def check_layers(model, tokens, prefill_n, layers, log=print):
    """Each layer on the reference's inputs: prefill of the first prefill_n positions (a multiple
    of 32), then decode of every later position. Returns {(layer, "prefill" or position): (pcc,
    largest difference)}."""
    import ttnn

    ref = reference(model.export, tokens, layers)
    out = {}
    for i in layers:
        layer = model.layers[i]
        x, y = ref[i]
        t0 = time.monotonic()
        cos, sin = model.rope.prefill(0, prefill_n)
        got = ttnn.to_torch(layer.prefill(to_device(model.device, x[:prefill_n]), cos, sin))
        got = got.reshape(-1, got.shape[-1])[:prefill_n].float()
        out[(i, "prefill")] = (pcc(got, y[:prefill_n]), float((got - y[:prefill_n]).abs().max()))
        log(
            f"layer {i} prefill {prefill_n}: pcc {out[(i, 'prefill')][0]:.6f}, "
            f"max diff {out[(i, 'prefill')][1]:.4g} ({time.monotonic() - t0:.0f} s)"
        )
        for p in range(prefill_n, len(tokens)):
            t0 = time.monotonic()
            cur, idx = model.position(p)
            cos, sin = model.rope.decode(idx, model.mcs["rope"])
            xd = to_device(model.device, x[p : p + 1], logical=1)
            got = ttnn.to_torch(layer.decode(xd, cos, sin, cur, model.mcs))
            got = got.reshape(-1, got.shape[-1])[:1].float()
            out[(i, p)] = (pcc(got, y[p : p + 1]), float((got - y[p : p + 1]).abs().max()))
            log(
                f"layer {i} decode at {p}: pcc {out[(i, p)][0]:.6f}, "
                f"max diff {out[(i, p)][1]:.4g} ({time.monotonic() - t0:.0f} s)"
            )
    return out
