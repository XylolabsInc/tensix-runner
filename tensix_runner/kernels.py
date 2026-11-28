# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Xylolabs
"""The optional tensix-kernels path: the dense reader (a weight-streaming matmul launched through
ttnn.generic_op) for the model's matrices.

Off unless enabled: `--kernels PATH` on the command line or `TENSIX_KERNELS=PATH` in the
environment, PATH a checkout of https://github.com/XylolabsInc/tensix-kernels. Without it every
matrix is a ttnn tensor and ttnn.linear runs it.

With it, each matrix is stored once, DRAM width-sharded over the device's banks, and the dense
reader runs it for decode and prefill (prefill in its row slices). The values are the export's: a
matrix on its BFP grid is packed without moving a value. The LM head uses lm_head_topk's device head and picks the final token on the device. On ttsim's Blackhole the dense
reader at HiFi4 gives ttnn.linear's output at HiFi4 bit for bit at 1 and 32 rows; on the Wormhole
it accumulates in fp32 where ttnn.linear here accumulates in bf16, so it is the closer of the two."""

import os
import sys
from pathlib import Path

HEAD_ROWS = 8  # the head's rows a call (rows * k <= 64 per core, with k = 8)

_STATE = {"dr": None, "path": None}


def enable(path=None):
    """Load the dense reader from a tensix-kernels checkout (path, else TENSIX_KERNELS). Returns
    whether the kernels are on."""
    path = path or os.environ.get("TENSIX_KERNELS")
    if not path:
        return False
    root = Path(path).expanduser().resolve()
    if not (root / "dense_reader" / "linear.py").exists():
        raise FileNotFoundError(
            f"{root}: not a tensix-kernels checkout (no dense_reader/linear.py)"
        )
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from dense_reader import linear as dr

    _STATE.update(dr=dr, path=root)
    return True


def enabled():
    return _STATE["dr"] is not None


def where():
    """The tensix-kernels checkout in use, or None."""
    return _STATE["path"]


class Weight:
    """A PyTorch [out, in] matrix as a dense-reader weight [1, 1, in, out] at fmt, DRAM
    width-sharded (the out dimension padded to whole tiles a bank where the bank count needs it)."""

    def __init__(self, device, w, fmt):
        import ttnn

        dtype = {"bf16": ttnn.bfloat16, "bfp8": ttnn.bfloat8_b, "bfp4": ttnn.bfloat4_b}[fmt]
        self.n = w.shape[0]
        self.t, _ = _STATE["dr"].native_weight(device, w.float(), dtype)

    def __call__(self, x, fidelity, mc):
        import ttnn

        y = _STATE["dr"].matmul(x, self.t, None, fidelity=fidelity, memory_config=mc)
        if y.shape[-1] != self.n:  # padding columns of a bank count that does not divide out
            z = ttnn.slice(y, [0, 0, 0, 0], [1, 1, y.shape[-2], self.n], memory_config=mc)
            ttnn.deallocate(y)
            y = z
        return y


class Head:
    """lm_head_topk's DeviceHead over the export's bf16 head: the dense reader's top-k records over
    vocabulary blocks of the head's own bf16 values (each core's best 8 of each row), then on the
    device an exact rescore of each row's best 16 candidates (bf16 rows, float32 sums) and the
    pick, the lowest id on ties: the argmax of float32 logits. table: the head's rows already on
    the device in ttnn.embedding's layout (a tied embedding), else the head makes its own copy. Up
    to HEAD_ROWS (8) rows a call."""

    def __init__(self, device, w, table=None, chunks=16):
        import torch
        from lm_head_topk import lmhead

        vocab, dim = w.shape
        step = -(-vocab // (chunks * 32)) * 32  # vocabulary blocks of whole tiles
        padded = step * chunks
        if padded != vocab:  # pad rows: never candidates (valid)
            w = torch.cat([w, torch.zeros(padded - vocab, dim, dtype=w.dtype)])
            table = None  # the pick checks that a given table has every padded row
        self.head = lmhead.DeviceHead(
            device,
            w.to(torch.bfloat16),
            fmt="bf16",
            chunks=chunks,
            k=8,  # rows * k <= 64 per core: HEAD_ROWS rows a call (k=4 stops ttsim, b2f72be)
            top=16,
            logits="float32",
            device_pick=True,
            rows=table,
            valid=vocab if padded != vocab else None,
        )

    def __call__(self, h, rows):
        """The ids of h's first rows (h normed, [1, 1, >= rows, dim] bf16 TILE), up to HEAD_ROWS."""
        if rows > HEAD_ROWS:
            raise ValueError(f"{rows} rows: the head takes at most {HEAD_ROWS} rows a call")
        return [int(t) for t in self.head(h, rows)]
