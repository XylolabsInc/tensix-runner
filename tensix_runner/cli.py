# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Xylolabs
"""tensix-runner's command line.

tensix-runner download NAME [--repo ORG/NAME] [--dir DIR]   fetch an export from Hugging Face
tensix-runner info [--device-id N]                          what ttnn reports about the device
tensix-runner check EXPORT [--host | --device] [--layers 0,1]
    every projection quant4.json lists packs at its format without moving a value: --host with
    torch alone (no ttnn), by default through ttnn's packer, with --device also through DRAM
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
    a = ap.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
