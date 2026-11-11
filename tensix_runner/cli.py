# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Xylolabs
"""tensix-runner's command line.

tensix-runner download NAME [--repo ORG/NAME] [--dir DIR]   fetch an export from Hugging Face
tensix-runner info [--device-id N]                          what ttnn reports about the device
tensix-runner check EXPORT [--host | --device] [--layers 0,1]
    every projection quant4.json lists packs at its format without moving a value: --host with
    torch alone (no ttnn), by default through ttnn's packer, with --device also through DRAM
tensix-runner pcc EXPORT [--layers 0,17] [--prefill 32 --decode 2] [--greedy 8]
    device layers on transformers' inputs, and greedy tokens, against transformers on the CPU
tensix-runner run EXPORT PROMPT [--max-new 256] [--raw] [--think]   one prompt, greedy
tensix-runner chat EXPORT [--system TEXT]                           a conversation on the terminal
tensix-runner bench EXPORT [--pp 512] [--tg 128] [--depth 0]        llama-bench's pp and tg, timed here
"""

import argparse
import sys
import time
from pathlib import Path

from . import HF_NAMESPACE


def ints(s):
    return {int(x) for x in s.split(",") if x != ""}


def cmd_download(a):
    from huggingface_hub import snapshot_download

    repo = a.repo or f"{HF_NAMESPACE}/{a.name}"
    out = Path(a.dir or Path("exports") / repo.split("/")[-1])
    path = snapshot_download(repo, local_dir=out)
    print(f"{repo} -> {path}")
    return 0


def cmd_info(a):
    from importlib.metadata import version

    import ttnn

    from .device import info, open_device

    dev = open_device(a.device_id)
    try:
        print(f"ttnn {version('ttnn')}: {info(dev)}")
    finally:
        ttnn.close_device(dev)
    return 0


def cmd_check(a):
    from .export import Export

    ex = Export(a.export)
    layers = ints(a.layers) if a.layers else None
    dev = None
    if a.host:
        from .bfp import on_grid

        def ok(w, fmt):
            return on_grid(w, fmt)
    else:
        from . import weights

        if a.device:
            from .device import info, open_device

            dev = open_device(a.device_id)
            print(f"device: {info(dev)}", flush=True)

        def ok(w, fmt):
            return weights.check(w, fmt, dev)

    t0, n, bad = time.monotonic(), 0, []
    try:
        for layer, proj, fmt, name in ex.matrices(layers):
            if fmt == "bf16" and not a.host:
                continue  # bf16 packs as it is stored
            n += 1
            if not ok(ex.get(name), fmt):
                bad.append(f"layer {layer} {proj} ({fmt})")
                print(f"MOVED layer {layer} {proj} ({fmt})", flush=True)
    finally:
        if dev is not None:
            import ttnn

            ttnn.close_device(dev)
        ex.close()
    how = "torch's grid check" if a.host else ("ttnn through DRAM" if a.device else "ttnn's packer")
    print(
        f"{ex.root.name}: {n - len(bad)} of {n} matrices exact ({how}) in {time.monotonic() - t0:.0f} s"
    )
    return 1 if bad else 0


def cmd_pcc(a):
    import ttnn

    from . import pcc
    from .device import info, open_device
    from .export import Export
    from .models import load

    ex = Export(a.export)
    layers = sorted(ints(a.layers)) if a.layers else None
    n = a.prefill + a.decode
    tokens = pcc.prompt_tokens(ex, n)
    dev = open_device(a.device_id)
    bad = []
    try:
        print(f"device: {info(dev)}", flush=True)
        t0 = time.monotonic()
        # the whole model for greedy tokens, else only the layers compared (a verify check takes
        # --layers as the model's layers, with the head)
        model = load(
            ex,
            dev,
            max_seq=max(256, -(-(n + a.verify + a.spec + a.k + 1) // 32) * 32),
            layers=None if a.greedy or ((a.verify or a.spec) and not layers) else layers or [],
            head=a.greedy > 0 or a.verify > 0 or a.spec > 0,
        )
        print(f"loaded in {time.monotonic() - t0:.0f} s", flush=True)
        overrides = None
        if a.window:  # both sides: the model and the reference
            model.cfg.window = a.window
            overrides = {"sliding_window": a.window}
        if a.spec:
            from . import generate as g
            from .models.qwen35 import Mtp

            mtp = Mtp(model)
            prompt = tokens[: a.prefill]
            plain, _, t_plain = g.generate(model, prompt, a.spec)
            spec, _, t_spec, kept = g.speculate(model, mtp, prompt, a.spec, a.k)
            same = sum(x == y for x, y in zip(plain, spec))
            print(
                f"spec k={a.k}: {same} of {a.spec} tokens equal to plain; kept a round {kept}"
                f"\n plain {plain}\n spec  {spec}\n (plain {t_plain:.0f} s, spec {t_spec:.0f} s)",
                flush=True,
            )
            if plain != spec:
                bad.append("spec")
        elif a.verify:
            more = pcc.prompt_tokens(ex, a.prefill + a.verify + 1)
            keep = max(1, a.verify - 1)
            ver, seq, after = pcc.check_verify(
                model, more, a.prefill, a.verify, keep, log=lambda s: print(s, flush=True)
            )
            if ver != seq or after != seq[keep]:
                bad.append("verify")
        elif layers:
            res = pcc.check_layers(
                model, tokens, a.prefill, layers, lambda s: print(s, flush=True), overrides
            )
            bad += [k for k, (r, _) in res.items() if r < a.bar]
        if a.greedy:
            got = [model.prefill(tokens[: a.prefill])]
            for k in range(a.greedy - 1):
                got.append(model.decode(got[-1], a.prefill + k))
            want = pcc.greedy_reference(ex, tokens[: a.prefill], a.greedy, overrides)
            same = sum(g == w for g, w in zip(got, want))
            print(f"greedy: {same} of {a.greedy} tokens equal\n device {got}\n ref    {want}")
            if got != want:
                bad.append("greedy")
    finally:
        ttnn.close_device(dev)
    print(f"{len(bad)} below the bar {a.bar}" + (f": {bad}" if bad else ""))
    return 1 if bad else 0


class Stream:
    """Prints the decoded text of the tokens so far as each new token arrives."""

    def __init__(self, tok, stop):
        self.tok, self.stop, self.ids, self.shown = tok, stop, [], ""

    def __call__(self, t):
        if t in self.stop:
            return
        self.ids.append(t)
        text = self.tok.decode(self.ids)
        if not text.endswith("\ufffd"):  # wait for the rest of a multi-byte character
            print(text[len(self.shown) :], end="", flush=True)
            self.shown = text


def _open_model(a):
    from .device import info, open_device
    from .export import Export
    from .models import load

    ex = Export(a.export)
    trace = getattr(a, "trace", False)
    dev = open_device(a.device_id, trace_region_size=a.trace_mb << 20 if trace else 0)
    print(f"device: {info(dev)}", flush=True)
    t0 = time.monotonic()
    model = load(ex, dev, max_seq=a.max_seq, trace=trace)
    print(f"{ex.root.name} loaded in {time.monotonic() - t0:.0f} s", flush=True)
    return ex, dev, model


def _summary(out, t_pf, t_dec, n_prompt, trace=False):
    dec = len(out) - 1
    rate = f", decode {dec / t_dec:.2f} tok/s" if dec and t_dec > 0 else ""
    how = "traced decode" if trace else "eager, no trace"
    return f"[{n_prompt} prompt tokens in {t_pf:.2f} s, {len(out)} new{rate}; {how}]"


def cmd_run(a):
    import ttnn

    from . import generate as g

    ex, dev, model = _open_model(a)
    try:
        tok, stop = g.tokenizer(ex), g.stop_ids(ex)
        text = (
            a.prompt
            if a.raw
            else g.chat_prompt(ex, [{"role": "user", "content": a.prompt}], thinking=a.think)
        )
        ids = tok.encode(text, add_special_tokens=False).ids
        if a.raw:  # starts as the template would (BOS for Gemma)
            ids = g.template_bos(ex, tok) + ids
        out, t_pf, t_dec = g.generate(model, ids, a.max_new, stop, Stream(tok, stop))
        print("\n" + _summary(out, t_pf, t_dec, len(ids), a.trace), flush=True)
    finally:
        ttnn.close_device(dev)
    return 0


def cmd_chat(a):
    import ttnn

    from . import generate as g

    ex, dev, model = _open_model(a)
    msgs = [{"role": "system", "content": a.system}] if a.system else []
    try:
        tok, stop = g.tokenizer(ex), g.stop_ids(ex)
        while True:
            try:
                line = input("> ")
            except EOFError:
                break
            if not line.strip():
                continue
            msgs.append({"role": "user", "content": line})
            ids = tok.encode(
                g.chat_prompt(ex, msgs, thinking=a.think), add_special_tokens=False
            ).ids
            s = Stream(tok, stop)
            out, t_pf, t_dec = g.generate(model, ids, a.max_new, stop, s)
            print("\n" + _summary(out, t_pf, t_dec, len(ids), a.trace), flush=True)
            msgs.append({"role": "assistant", "content": s.shown})
    finally:
        ttnn.close_device(dev)
    return 0


def cmd_bench(a):
    """pp and tg in llama-bench's form, one user, timed on this device (each pp from position 0;
    tg as tg_n decode steps after a pp of the depth)."""
    import statistics

    import ttnn

    from . import generate as g

    ex, dev, model = _open_model(a)
    try:
        tok = g.tokenizer(ex)
        text = " ".join(["The quick brown fox jumps over the lazy dog."] * 400)
        ids = tok.encode(text, add_special_tokens=False).ids
        model.prefill(ids[:32])  # compile the programs once, untimed
        print("| test | depth | tok/s | runs |\n|---|---:|---:|---:|")
        for n in ints(a.pp):
            rates = []
            for _ in range(a.repeat):
                t0 = time.perf_counter()
                model.prefill(ids[:n])
                rates.append(n / (time.perf_counter() - t0))
            print(f"| pp{n} | 0 | {statistics.mean(rates):.2f} | {a.repeat} |", flush=True)
        for d in ints(a.depth):
            rates = []
            for _ in range(a.repeat):
                t = model.prefill(ids[: max(d, 1)])
                t0 = time.perf_counter()
                for k in range(a.tg):
                    t = model.decode(t, max(d, 1) + k)
                rates.append(a.tg / (time.perf_counter() - t0))
            print(f"| tg{a.tg} | {d} | {statistics.mean(rates):.2f} | {a.repeat} |", flush=True)
        print(("traced decode" if a.trace else "eager (no trace)") + "; one user")
    finally:
        ttnn.close_device(dev)
    return 0


def _trace_args(p):
    p.add_argument(
        "--trace",
        action="store_true",
        help="replay decode steps from a trace (a card only: ttsim has none; not yet run on a card)",
    )
    p.add_argument("--trace-mb", type=int, default=64, help="the device's trace region, MiB")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="tensix-runner", description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("download", help="fetch an export from Hugging Face")
    p.add_argument("name", help="the export's name, e.g. Qwen3-4B-Tensix-BFP4")
    p.add_argument("--repo", help=f"the full repo id (default {HF_NAMESPACE}/NAME)")
    p.add_argument("--dir", help="where to put it (default exports/NAME)")
    p.set_defaults(fn=cmd_download)
    p = sub.add_parser("info", help="what ttnn reports about the device")
    p.add_argument("--device-id", type=int, default=0)
    p.set_defaults(fn=cmd_info)
    p = sub.add_parser("check", help="every matrix packs without moving a value")
    p.add_argument("export", help="the export's directory")
    g = p.add_mutually_exclusive_group()
    g.add_argument("--host", action="store_true", help="torch alone, no ttnn")
    g.add_argument("--device", action="store_true", help="also write each matrix to DRAM and back")
    p.add_argument("--device-id", type=int, default=0)
    p.add_argument("--layers", help="comma-separated layer numbers (default: all)")
    p.set_defaults(fn=cmd_check)
    p = sub.add_parser("pcc", help="layers and greedy tokens against transformers (the ref extra)")
    p.add_argument("export", help="the export's directory")
    p.add_argument("--layers", help="comma-separated layers to compare one by one")
    p.add_argument("--prefill", type=int, default=32, help="prompt rows (a multiple of 32)")
    p.add_argument("--decode", type=int, default=2, help="decode steps after the prompt")
    p.add_argument("--greedy", type=int, default=0, help="whole model: this many greedy tokens")
    p.add_argument(
        "--spec",
        type=int,
        default=0,
        help="MTP drafting: this many tokens drafted and verified against plain greedy (with --layers: those only)",
    )
    p.add_argument("--k", type=int, default=2, help="--spec: drafts a round")
    p.add_argument(
        "--verify",
        type=int,
        default=0,
        help="a verify of this many teacher-forced rows against decode steps (with --layers: those only)",
    )
    p.add_argument(
        "--window",
        type=int,
        default=0,
        help="a sliding window for this check, set on both sides (reaches the window's edge in a short prompt)",
    )
    p.add_argument("--bar", type=float, default=0.99, help="the PCC every output must reach")
    p.add_argument("--device-id", type=int, default=0)
    p.set_defaults(fn=cmd_pcc)
    for name, fn, what in (
        ("run", cmd_run, "one prompt, greedy, printed as it is decoded"),
        ("chat", cmd_chat, "a conversation on the terminal (each turn prefills the whole history)"),
    ):
        p = sub.add_parser(name, help=what)
        p.add_argument("export", help="the export's directory")
        if name == "run":
            p.add_argument("prompt")
            p.add_argument("--raw", action="store_true", help="the prompt as is, no chat template")
        else:
            p.add_argument("--system", help="a system message")
        p.add_argument(
            "--think", action="store_true", help="the template's thinking mode, where it has one"
        )
        p.add_argument("--max-new", type=int, default=256)
        p.add_argument("--max-seq", type=int, default=2048, help="the KV cache's length")
        _trace_args(p)
        p.add_argument("--device-id", type=int, default=0)
        p.set_defaults(fn=fn)
    p = sub.add_parser("bench", help="pp and tg in llama-bench's form, timed on this device")
    p.add_argument("export", help="the export's directory")
    p.add_argument("--pp", default="512", help="prompt lengths, comma-separated")
    p.add_argument("--tg", type=int, default=128, help="decode steps per tg run")
    p.add_argument("--depth", default="0", help="tg depths, comma-separated")
    p.add_argument("--repeat", type=int, default=3)
    p.add_argument("--max-seq", type=int, default=4096, help="the KV cache's length")
    _trace_args(p)
    p.add_argument("--device-id", type=int, default=0)
    p.set_defaults(fn=cmd_bench)
    a = ap.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
