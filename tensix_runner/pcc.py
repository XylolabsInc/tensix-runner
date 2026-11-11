# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Xylolabs
"""Layer and model checks against transformers on the same export (the `ref` extra).

The reference runs the export in float32 on the CPU and records every decoder layer's input and
output at every position. For each checked layer, the device layer gets the reference's input rows:
a prefill over the first T positions, then one decode step per later position on the KV rows the
prefill wrote. Each output is compared with the reference's by the Pearson correlation (PCC) and
the largest absolute difference. The whole-model check compares greedy tokens step by step, the
device decoding transformers' tokens."""

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
    """transformers' greedy continuation (n tokens) in float32 on the CPU, and each step's logits
    [vocab] before any final softcap (Gemma's tanh(x / cap) * cap is undone)."""
    m = _hf_model(export, overrides)
    with torch.no_grad():
        out = m.generate(
            torch.tensor([tokens]),
            max_new_tokens=n,
            do_sample=False,
            output_logits=True,
            return_dict_in_generate=True,
        )
    cap = export.text_config.get("final_logit_softcapping")
    logits = []
    for row in out.logits:
        x = row[0].double()
        logits.append(cap * torch.atanh(x / cap) if cap else x)
    return out.sequences[0, len(tokens) :].tolist(), logits


def bf16_ulp(x):
    """The spacing of bfloat16 values at x (8 significant bits)."""
    e = torch.floor(torch.log2(torch.tensor(abs(float(x)) or 1e-30, dtype=torch.float64)))
    return float(2.0 ** (e - 7))


def compare_greedy(got, want, logits):
    """Step by step, the device's token against transformers' (the device decoding transformers'
    tokens, so one difference does not shift the rest). A difference counts as a tie when the two
    tokens' logits in transformers lie within one bfloat16 step of each other. Returns
    (equal steps, ties [(step, device, transformers, gap)], other differences [same])."""
    same, ties, wrong = 0, [], []
    for k, (g, w) in enumerate(zip(got, want)):
        if g == w:
            same += 1
            continue
        gap = float(logits[k][w] - logits[k][g])
        (ties if gap <= bf16_ulp(logits[k][w]) else wrong).append((k, g, w, round(gap, 4)))
    return same, ties, wrong


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


def check_drift(model, tokens, n, log=print, overrides=None):
    """The whole model's prefill of the first n tokens (a multiple of 32), each layer on the
    previous device layer's output: every layer's output against transformers' at the same layer,
    so that the drift through the depth shows. Returns {layer: (pcc, largest difference)}."""
    import ttnn

    ref = reference(model.export, tokens[:n], sorted(model.layers), overrides)
    got = {}
    for i, layer in model.layers.items():

        def wrap(*args, _i=i, _f=layer.prefill, **kw):
            y = _f(*args, **kw)
            t = ttnn.to_torch(y).float()
            got[_i] = t.reshape(-1, t.shape[-1])[:n]
            return y

        layer.prefill = wrap
    t0 = time.monotonic()
    try:
        model.prefill(tokens[:n])
    finally:
        for layer in model.layers.values():
            del layer.prefill  # back to the class's method
    out = {}
    for i in sorted(got):
        y = ref[i][1][:n]
        out[i] = (pcc(got[i], y), float((got[i] - y).abs().max()))
        log(f"layer {i} in the whole prefill: pcc {out[i][0]:.6f}, max diff {out[i][1]:.4g}")
    log(f"({time.monotonic() - t0:.0f} s)")
    return out


def mtp_reference(export, tokens, n):
    """Qwen3.5's MTP built from transformers' parts, float32 on the CPU (transformers does not run
    the MTP itself, so this follows vLLM's Qwen3-Next MTP): the model's last-layer rows H for
    tokens[:n] (before its final norm); fc over [norm_e(embedding of tokens[1 : n + 1]);
    norm_h(final_norm(H))]; one Qwen3_5DecoderLayer of full attention
    holding the export's mtp.layers.0 weights, at the positions and with the RoPE the model used;
    mtp.norm; the model's head. Returns (H [n, dim], the normed MTP rows [n, dim], logits [n, vocab])."""
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5DecoderLayer, Qwen3_5RMSNorm

    m = _hf_model(export)
    m.eval()
    inner = m.model
    lm = inner.language_model if hasattr(inner, "language_model") else inner
    tc = lm.config
    full = tc.layer_types.index("full_attention")
    rec = {}

    def keep_rope(mod, args, kwargs):
        rec["rope"] = kwargs["position_embeddings"]

    def keep_last(mod, args, kwargs, out):
        rec["h"] = (out[0] if isinstance(out, tuple) else out).detach().clone()

    hs = [
        lm.layers[full].register_forward_pre_hook(keep_rope, with_kwargs=True),
        lm.layers[-1].register_forward_hook(keep_last, with_kwargs=True),
    ]
    with torch.no_grad():
        m(input_ids=torch.tensor([tokens[:n]]))
    for h in hs:
        h.remove()

    def get(name):
        return export.get(name).float()

    layer = Qwen3_5DecoderLayer(tc, full).float().eval()
    pre = "mtp.layers.0."
    names = [k for k in export.weight_map if k.startswith(pre)]
    layer.load_state_dict({k[len(pre) :]: get(k) for k in names}, strict=True)
    norms = {}
    for key in ("pre_fc_norm_embedding", "pre_fc_norm_hidden", "norm"):
        norms[key] = Qwen3_5RMSNorm(tc.hidden_size, eps=tc.rms_norm_eps).float()
        norms[key].weight.data = get(f"mtp.{key}.weight")
    with torch.no_grad():
        hidden = rec["h"]
        e = lm.embed_tokens(torch.tensor([tokens[1 : n + 1]]))
        x = torch.cat(
            [norms["pre_fc_norm_embedding"](e), norms["pre_fc_norm_hidden"](lm.norm(hidden))],
            dim=-1,
        )
        x = x @ get("mtp.fc.weight").T
        y = layer(x, position_embeddings=rec["rope"], position_ids=torch.arange(n)[None])
        y = norms["norm"](y)
        logits = m.lm_head(y)
    return hidden[0], y[0], logits[0]


def check_mtp(model, mtp, tokens, n, log=print):
    """The device MTP (mtp.rows, a prompt of n rows from position 0) on transformers' last-layer
    rows against mtp_reference: the normed rows' PCC and every row's draft against the reference's
    argmax, a difference within one bfloat16 step counting as a tie. Returns (pcc, largest
    difference, equal rows, ties, other differences)."""
    import ttnn

    assert 1 < n <= 32 and n + 1 <= len(tokens)
    h, want_y, logits = mtp_reference(model.export, tokens, n)
    t0 = time.monotonic()
    y, _ = mtp.rows(to_device(model.device, h), tokens[1 : n + 1], 0)
    got_y = ttnn.to_torch(y).float().reshape(-1, h.shape[-1])[:n]
    got = model.greedy_normed(y, n)
    ttnn.deallocate(y)
    want = logits.argmax(-1).tolist()
    same, ties, wrong = compare_greedy(got, want, [row.double() for row in logits])
    r = (pcc(got_y, want_y), float((got_y - want_y).abs().max()))
    log(
        f"mtp rows 0-{n - 1}: pcc {r[0]:.6f}, max diff {r[1]:.4g}; drafts {same} of {n} equal, "
        f"{len(ties)} ties {ties}, {len(wrong)} other differences {wrong} "
        f"({time.monotonic() - t0:.0f} s)"
    )
    return (*r, same, ties, wrong)


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
