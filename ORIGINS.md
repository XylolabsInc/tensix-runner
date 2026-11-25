# Origins

Where every file of this repository comes from, and how that was checked.

## Classes

- **ours**: written for this repository.
- **ours, follows tt-metal**: its structure follows a named tt-metal file (https://github.com/tenstorrent/tt-metal, Apache-2.0), so it is treated as a
  modified version of that file: it carries that file's Tenstorrent SPDX lines above ours, and [NOTICE](NOTICE) names
  the file. No text is copied beyond calls of tt-metal's public APIs.
- **adapted**: taken from a public repository under its declared license (the source, license and changes are in the
  "Source" and "Notes" columns and in the last section); it keeps that source's copyright line beside ours.

One file is adapted, from Xylolabs' own tensix-bfp-gptq (Apache-2.0). No file follows a tt-metal file. The code
calls the public APIs of ttnn, torch, safetensors, tokenizers, huggingface_hub, jinja2 and numpy.

## Files

One row per tracked file, except `LICENSE` (the text of the Apache License, Version 2.0), `NOTICE`, this file and
`DEVELOPMENT.md`; `tests/test_origins.py` checks that each appears exactly once. "Source" names the file a row follows
or is adapted from. "Lines checked" counts the lines of 30 or more characters (whitespace-normalized), every line
counted, duplicates included; "Lines in tt-metal" counts those also found in the tt-metal tree described below.

| File | Class | Source | Lines checked | Lines in tt-metal | Notes |
|---|---|---|---|---|---|
| `.gitignore` | ours |  | 2 | 0 | ignore patterns |
| `AGENTS.md` | ours |  | 45 | 0 | instructions for coding agents |
| `CLAUDE.md` | ours |  | 0 | 0 | symbolic link to AGENTS.md |
| `README.md` | ours |  | 77 | 0 | documentation |
| `pyproject.toml` | ours |  | 20 | 0 | package metadata |
| `scripts/download.sh` | ours |  | 5 | 1 |  |
| `scripts/install.sh` | ours |  | 35 | 1 |  |
| `scripts/ttsim.sh` | ours |  | 11 | 4 |  |
| `tensix_runner/__init__.py` | ours |  | 3 | 1 | tensix-runner: Xylolabs' BFP4/BFP8 Hugging Face exports on Tenstorrent Tensix devices through ttnn |
| `tensix_runner/__main__.py` | ours |  | 1 | 1 |  |
| `tensix_runner/bfp.py` | adapted | Xylolabs' tensix-bfp-gptq, `tensix_bfp/bfp.py` at `d67a977`, Apache-2.0 | 24 | 1 | `bfp()` as is; `on_grid()` reduced to the formats the runner loads |
| `tensix_runner/cli.py` | ours |  | 171 | 8 | tensix-runner's command line |
| `tensix_runner/device.py` | ours |  | 25 | 3 |  |
| `tensix_runner/export.py` | ours |  | 74 | 3 |  |
| `tensix_runner/generate.py` | ours |  | 59 | 2 | Prompts, stop tokens and the greedy loop |
| `tensix_runner/kernels.py` | ours |  | 48 | 1 | The optional tensix-kernels path: the dense reader (a weight-streaming matmul launched through ttnn.generic_op) for the model's matrices |
| `tensix_runner/kv.py` | ours |  | 55 | 7 | A layer's paged KV cache for one sequence |
| `tensix_runner/models/__init__.py` | ours |  | 8 | 1 | The model families, by the checkpoint's model_type |
| `tensix_runner/models/gemma4.py` | ours |  | 213 | 15 | Gemma 4's text models with per-layer inputs (gemma-4-E2B, gemma-4-E4B) on one chip |
| `tensix_runner/models/qwen3.py` | ours |  | 191 | 18 | Qwen3 dense (Qwen3-4B, Qwen3-8B) on one chip |
| `tensix_runner/models/qwen35.py` | ours |  | 335 | 11 | Qwen3.5 (Qwen3.5-4B, Qwen3.5-9B) on one chip: Gated DeltaNet layers and gated attention layers |
| `tensix_runner/ops.py` | ours |  | 65 | 8 | Small building blocks on ttnn shared by the models: compute configs, matrices, norms and the MLP |
| `tensix_runner/pcc.py` | ours |  | 155 | 6 | Layer and model checks against transformers on the same export (the `ref` extra) |
| `tensix_runner/rope.py` | ours |  | 36 | 2 | Rotary position embedding in Hugging Face's layout (rotate_half: the first and second halves of each head pair up), for ttnn.experimental.rotary_embedding_hf |
| `tensix_runner/weights.py` | ours |  | 17 | 2 | Weights packed for TT-NN exactly as the export stores them |
| `tests/conftest.py` | ours |  | 36 | 5 | Shared fixtures |
| `tests/test_bfp.py` | ours |  | 21 | 2 | The torch grid check: bfp() on hand-computed blocks and on_grid() on rounded and raw weights |
| `tests/test_device.py` | ours |  | 18 | 2 |  |
| `tests/test_export.py` | ours |  | 100 | 4 | Export loading on synthetic exports: formats, prefixes, the head's name and tie rules, refusals, and the host check |
| `tests/test_export_device.py` | ours |  | 13 | 1 |  |
| `tests/test_gemma4.py` | ours |  | 39 | 9 | Gemma 4 on the host: which layer's KV cache each layer reads, and the RoPE frequencies of both layer types (against transformers where it is installed) |
| `tests/test_generate.py` | ours |  | 34 | 1 | The chat template, the stop tokens and the greedy loop's bookkeeping, on the host |
| `tests/test_kernels.py` | ours |  | 33 | 2 |  |
| `tests/test_kv.py` | ours |  | 57 | 10 |  |
| `tests/test_ops.py` | ours |  | 10 | 2 | compute_config's matmul accumulation: bf16 on ttsim's Wormhole only, fp32 everywhere else (needs the ttnn package, no device) |
| `tests/test_origins.py` | ours |  | 21 | 4 | every tracked file is listed in ORIGINS.md exactly once |
| `tests/test_pcc.py` | ours |  | 11 | 1 | The greedy check's bookkeeping on the host: bfloat16's spacing and the tie rule |
| `tests/test_qwen3.py` | ours |  | 14 | 2 |  |
| `tests/test_qwen35.py` | ours |  | 31 | 4 |  |

## How it was checked

All counts are from a run on 2026-11-25 over the files of this tree (on `67a2ab6` with this change), against the version
this repository pins, tt-metal v0.81.1.

1. **Line comparison.** Every line of 30 or more characters (whitespace-normalized), every line counted and duplicates
   included, of each file in the table was looked up in tt-metal v0.81.1 (its source checkout at tag `v0.81.1`,
   24,522 text files, plus the runtime tree of the `ttnn` 0.81.1 wheel, 6,665) and in
   transformers v5.13.0 (5,909), the GPTQ reference implementation
   [IST-DASLab/gptq](https://github.com/IST-DASLab/gptq) (commit `2d65066`, 41),
   [AutoGPTQ](https://github.com/AutoGPTQ/AutoGPTQ) (commit `9f7d370`, 172) and
   [llama.cpp](https://github.com/ggml-org/llama.cpp) (commit `d3c3e24`, 4,152). Of 2,113 lines
   checked, 145 are found in tt-metal's tree and 5 only in one of the other four trees. The lines found
   are license headers, import lines, calls of public APIs and common idioms.
2. **Tokens.** Each of the 34 `.py`, `.cpp`, `.h` and `.sh` files was compared with every Python, C, C++, CUDA and
   shell file of the same trees as a set of distinct 10-token sequences (comments dropped, and docstrings for Python):
   at most 16.67 % of a file's distinct sequences occur in any one upstream file (the highest is `tensix_runner/__main__.py`, closest
   to `tt-metal/.github/bug_checker/run_bug_checker.py`; a file of 1 lines checked, so few sequences); 16 files reach 2.5 % or more. Among
   the files of class "ours" with 50 or more lines checked, the highest is 5.82 % (`tensix_runner/kv.py`).

## Third-party model code and configs

Adapted files, each with its upstream repository, revision and license.

| File | Upstream | Revision | License |
|---|---|---|---|
| `tensix_runner/bfp.py` | [XylolabsInc/tensix-bfp-gptq](https://github.com/XylolabsInc/tensix-bfp-gptq), `tensix_bfp/bfp.py` | `d67a977` | Apache-2.0 |

No model code or model configuration of another project is included.
