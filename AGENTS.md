# AGENTS.md

Instructions for AI coding agents (Codex, Claude Code and others) working in this repository.
`CLAUDE.md` is a symlink to this file.

## What this is

tensix-runner runs Xylolabs' Hugging Face exports (BFP4 and BFP8 checkpoints made with
[tensix-bfp-gptq](https://github.com/XylolabsInc/tensix-bfp-gptq)) on Tenstorrent Tensix devices,
Wormhole and Blackhole, through upstream ttnn. The README covers installing, checking and running.

## Layout

| Path | Content |
|---|---|
| `tensix_runner/export.py` | an export on disk: config, `quant4.json`'s formats, tensors read one at a time |
| `tensix_runner/bfp.py` | the BFP grids in torch, for checks without ttnn |
| `tensix_runner/weights.py` | packing at the export's format with `ttnn.as_tensor`, and the exactness check |
| `tensix_runner/device.py` | opening a chip; its arch, grid, DRAM banks and L1, read from the device |
| `tensix_runner/ops.py`, `rope.py` | compute configs, matrices, RMSNorm, the gated MLPs (SiLU, GELU tanh); HF-style RoPE tables on the device |
| `tensix_runner/kv.py` | the paged KV cache of an attention layer |
| `tensix_runner/kernels.py` | the optional tensix-kernels path (`--kernels`): the dense reader for the matrices, the device head |
| `tensix_runner/models/` | one module per family (`qwen3.py`, `qwen35.py`, `gemma4.py`), chosen by the checkpoint's `model_type` |
| `tensix_runner/generate.py` | the chat template (Jinja2, as transformers renders it), stop tokens, the greedy and MTP-drafted loops |
| `tensix_runner/pcc.py` | layer and greedy-token checks against transformers on the same export |
| `tensix_runner/cli.py` | `tensix-runner download`, `info`, `check`, `pcc`, `run`, `chat`, `bench` |
| `scripts/` | `install.sh` (venv, ttnn, SFPI, ttsim), `ttsim.sh` (the simulator's environment), `download.sh` |
| `tests/` | pytest; `device`-marked tests need ttnn and a card or ttsim, `export`-marked ones an export |

## Setup and checks

```
scripts/install.sh --ttsim bh        # or wh; Linux x86_64, Python 3.12
source .venv/bin/activate && source scripts/ttsim.sh bh
pytest                               # the device tests run on ttsim here
uvx ruff@0.17.4 check . && uvx ruff@0.17.4 format --check .
shellcheck -S warning scripts/*.sh
```

Without ttnn, `pytest` runs the host tests and skips the rest.

## Rules

- **Exact weights.** Every matrix is packed at the format `quant4.json` gives it, as `[in, out]`,
  with `ttnn.as_tensor`. Fuse matrices only along the output dimension at tile boundaries
  (multiples of 32). A new loader path needs a test that its packed values equal the stored ones.
- **Nothing computes on the host.** The embedding lookup, RoPE rows, the LM head, argmax and any
  drafter run on the device. The host tokenizes, schedules and copies token ids.
- **No device assumptions.** Read the grid, the DRAM bank count and L1 from the device
  (`device.info`). No card-specific constants. Matmuls on Wormhole accumulate in bf16 (`ops.compute_config`):
  ttsim's Wormhole stops on fp32 partial sums reloaded into its 19-bit source registers.
- **A new model** passes `tensix-runner pcc` (each layer at PCC 0.99 or better against transformers, prefill
  and decode) on ttsim's Blackhole and Wormhole before it is listed as working.
- **Upstream only.** Use the public ttnn API and public sources (tt-metal, ttnn, ttsim and
  Tenstorrent's docs). A file adapted from elsewhere keeps its copyright line and gets a row in
  `ORIGINS.md`. Nothing you can't publish under Apache-2.0.
- **Claims match what ran.** Say what ran where: ttsim (which simulator) or a named card. Never
  write a speed or accuracy figure that nobody measured.
- **Pinned stack.** ttnn 0.81.1, Python 3.12, ttsim v1.16.0, SFPI 7.89.0. Changing a pin means
  rerunning the device tests on it.

## Style

- Python 3.12; ruff 0.17.4 with line length 100 and sorted imports.
- American spelling in code, comments and docs.
- Plain, short docs: say what a thing does and how to run it.
- Commits: Conventional Commits with a gitmoji, imperative mood, header under 72 characters, for
  example `feat(export): ✨ read the tied head from the embedding`. One change per commit, signed.
