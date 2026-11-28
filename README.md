<div align="center">

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/XylolabsInc/tensix-bfp-gptq/main/assets/xylolabs-lockup-dark.svg">
  <img src="https://raw.githubusercontent.com/XylolabsInc/tensix-bfp-gptq/main/assets/xylolabs-lockup.svg" alt="Xylolabs" height="40">
</picture>

# tensix-runner

[![License](https://img.shields.io/badge/license-Apache--2.0-blue)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.12-3776AB)](pyproject.toml)
[![ttnn](https://img.shields.io/badge/ttnn-0.81.1-5B2C83)](https://github.com/tenstorrent/tt-metal/releases/tag/v0.81.1)
[![Hugging Face](https://img.shields.io/badge/Hugging%20Face-models-FFD21E)](https://huggingface.co/Xylolabs)

</div>

Runs Xylolabs' Hugging Face exports of LLMs, quantized with GPTQ to TT-NN's block floating point
formats `ttnn.bfloat4_b` (BFP4) and `ttnn.bfloat8_b` (BFP8) by
[tensix-bfp-gptq](https://github.com/XylolabsInc/tensix-bfp-gptq), on a Tenstorrent Wormhole or
Blackhole chip through the public ttnn API of [tt-metal](https://github.com/tenstorrent/tt-metal)
v0.81.1. The embedding lookup, the LM head and its argmax, and the drafter of models with multi-token
prediction run on the device; the host tokenizes and schedules.

[DEVELOPMENT.md](DEVELOPMENT.md) says how this repository is developed, from Tenstorrent's open software stack only; [ORIGINS.md](ORIGINS.md) says where every file comes from and how that was checked.

## Models

| Family | Exports | Chips |
|---|---|---|
| Qwen3 | Qwen3-4B, Qwen3-8B | Blackhole, Wormhole |
| Qwen3.5 (MTP drafting checked with `pcc --spec`) | Qwen3.5-4B, Qwen3.5-9B | Blackhole |
| Gemma 4 | gemma-4-E2B, gemma-4-E4B | Blackhole, Wormhole |

Tested on [ttsim](https://github.com/tenstorrent/ttsim) v1.16.0, Tenstorrent's simulator, not yet on
a card. `--trace` (decode replayed from a trace) needs a card and has not run yet.

## Install

Linux x86_64 and Python 3.12: ttnn 0.81.1 ships wheels for CPython 3.12 (and 3.10) on Linux x86_64
only.

```
git clone https://github.com/XylolabsInc/tensix-runner.git && cd tensix-runner
scripts/install.sh                 # .venv with tensix-runner, ttnn 0.81.1, torch (CPU), transformers, pytest and SFPI 7.89.0
source .venv/bin/activate
```

`install.sh` uses [uv](https://docs.astral.sh/uv/) when it is installed. Without uv it runs
`python3.12 -m venv`, so it needs a Python 3.12 whose `venv` module works (on Debian and Ubuntu,
the `python3.12-venv` package). The installer includes the `ref` and `test` extras. Use
`--venv DIR` to select a different environment directory.

`install.sh` unpacks SFPI 7.89.0, tt-metal v0.81.1's compiler for the Tensix cores, into ttnn's
`runtime` directory; `python -m ttnn.download_sfpi` does not work in 0.81.1. A card also needs Tenstorrent's driver and firmware
([tt-installer](https://github.com/tenstorrent/tt-installer)).

**Without a card,** ttsim stands in for one:

```
scripts/install.sh --ttsim bh      # or wh: libttsim and its SoC descriptor under ttsim/<arch>/
source scripts/ttsim.sh bh         # TT_METAL_SIMULATOR, slow dispatch
tensix-runner info                 # e.g. "blackhole (simulated): compute grid ..."
```

`source scripts/ttsim.sh bh|wh [DIR]` uses `ttsim/<arch>` by default. It sets
`TT_METAL_SIMULATOR` to the library path, `TT_METAL_SLOW_DISPATCH_MODE=1`, and
`TT_METAL_DISABLE_SFPLOADMACRO=1`. ttsim runs slow dispatch only, so traced decoding does not run on it.

## Get an export

```
tensix-runner download Qwen3-4B-Tenstorrent-Tensix-BFP4  # from Xylolabs/Qwen3-4B-Tenstorrent-Tensix-BFP4 into exports/
tensix-runner download NAME --repo ORG/NAME --dir DIR
tensix-runner download NAME --variant bfp4               # one folder of a repo with one export per format
```

A repo can have an export at its root or one export per format in folders such as `bfp4/` and
`bfp8/`. Use `--variant` when the repo has several export folders.
Root exports land in `exports/<repo name>` (or `--dir`), and folder exports land in their folder
under that directory.

`scripts/download.sh` accepts the same arguments; `PYTHON` selects its interpreter (default
`.venv/bin/python`).

`quant4.json` in the export names each projection's format.

The LM head is the tensor that `quant4.json`'s `lm_head_tensor` names, for a head stored under a
name such as `language_model.lm_head.weight`. If that field names a missing `lm_head.weight`
tensor (or is not a string), loading stops with an error; `null` counts as not declared. Without
the field the head is `lm_head.weight`, or else the embedding when `config.json` sets
`tie_word_embeddings` to true (at the top level or in `text_config`). A checkpoint with no head
tensor that does not say it ties its embedding fails with an error.

## Run

```
tensix-runner run exports/Qwen3-4B-Tenstorrent-Tensix-BFP4 "What is the capital of France?"
tensix-runner chat exports/Qwen3-4B-Tenstorrent-Tensix-BFP4 --system "You are concise."
tensix-runner bench exports/Qwen3-4B-Tenstorrent-Tensix-BFP4 --pp 512 --tg 128 --depth 0
```

`run` and `chat` render the checkpoint's own chat template (`--think` turns on thinking where the
template has it) and decode greedily, with the argmax on the device. `bench` times prefill (`pp`)
and decode (`tg`) in llama-bench's terms. `--trace` on any of them
replays the decode step from a trace (a card only); Qwen3.5 rejects traced decoding.
`run --raw` does not render the prompt with the chat template, but prepends a BOS token when
the template uses one. `--max-seq` must be a multiple of 32.

`--max-new 0` prefills nothing and generates no tokens; device, loading and summary messages still print; a negative `--max-new` is an error. `bench`
checks its lengths before it times anything: each `--pp` and each `--depth` must not exceed the
prompt tokens it has, and `--pp`, and `--depth` (at least 1) plus `--tg`, must fit `--max-seq`,
else it stops with a message naming the value and the limit. Its rates count the tokens actually
prefilled.

## Kernels (optional)

```
tensix-runner run exports/gemma-4-E2B-it-Tenstorrent-Tensix-BFP4-MLP "Hello" --kernels ../tensix-kernels
export TENSIX_KERNELS=../tensix-kernels      # applies to pcc, run, chat and bench
```

With a [tensix-kernels](https://github.com/XylolabsInc/tensix-kernels) checkout, every matrix
runs through its dense reader and the LM head through its device head, which picks the token on the
device. `--trace` does not take the kernels' head yet.

## Compare with transformers

```
tensix-runner pcc exports/Qwen3-4B-Tenstorrent-Tensix-BFP4 --layers 0,17,35 --prefill 32 --decode 2
tensix-runner pcc exports/Qwen3-4B-Tenstorrent-Tensix-BFP4 --greedy 8
tensix-runner pcc exports/gemma-4-E2B-it-Tenstorrent-Tensix-BFP4-MLP --drift
tensix-runner pcc exports/Qwen3.5-4B-Tenstorrent-Tensix-BFP4 --spec 8
```

`--layers` runs each listed layer on transformers' own inputs and prints its PCC and largest
difference (a layer that reads an earlier layer's KV cache brings that layer along). `--greedy`
has the device decode transformers' greedy tokens and compares every step's token; a difference
counts as a tie when transformers' two logits lie within one bfloat16 step. `--drift` compares
every layer's output within one whole-model prefill. `--window 32` sets
Gemma 4's sliding window on both sides.
`--greedy N` always compares exactly N steps: the reference does not stop at an end-of-sequence
token, the device decodes the reference's tokens (an end-of-sequence token is an ordinary token
there), and the sequence length covers the prompt plus N. `pcc` with none of `--layers`,
`--greedy`, `--verify`, `--spec`, `--drift` and `--mtp` checks nothing and stops with a usage
error before it opens a device; `--k` for `--spec` is an integer of at least 1, and at most 7 with
`--kernels` (a round sends k + 1 rows, and the kernels' head takes 8 rows a call).
`--mtp` checks Qwen3.5 MTP outputs against reference rows; `--spec N` compares N drafted
and verified tokens with plain greedy generation. `--verify N` compares teacher-forced verification
with decode steps.

transformers runs the export in float32 on the CPU. The weights alone need about twice the bf16
export size, with additional memory for loading and recorded activations.

## Check an export

```
tensix-runner check exports/Qwen3-4B-Tenstorrent-Tensix-BFP4 --host     # torch alone: every matrix on its grid
tensix-runner check exports/Qwen3-4B-Tenstorrent-Tensix-BFP4            # through ttnn's packer
tensix-runner check exports/Qwen3-4B-Tenstorrent-Tensix-BFP4 --device   # and through device DRAM and back
```

Each matrix is transposed to `[in, out]`, packed at its format and
unpacked again; the check passes when every value comes back bit for bit. Every matrix includes the
LM head, the tensor `lm_head_tensor` or the tie rules pick, at the format `quant4.json` gives it. A
bf16 matrix, head or projection, is checked with `--host` only. `--layers 0,1` limits the layers;
the head is always checked.

## Command options

All commands accept `-h` for help. `EXPORT` is an export directory; `NAME` identifies a download.

| Command | Options and defaults |
|---|---|
| `download NAME` | `--repo ORG/NAME` (default `Xylolabs/NAME`), `--dir DIR` (default `exports/<repo name>`), `--variant FOLDER` (select a folder export) |
| `info` | `--device-id 0` |
| `check EXPORT` | `--host` or `--device` (default: packer only), `--layers` (comma-separated, default all), `--device-id 0` |
| `pcc EXPORT` | `--layers` (default none), `--prefill 32` (positive; multiple of 32 for layers or drift), `--decode 2` (nonnegative), `--greedy 0`, `--mtp` (off), `--drift` (off), `--spec 0`, `--k 2`, `--verify 0`, `--window 0` (no override), `--bar 0.99`, `--device-id 0`, `--kernels PATH` (default `TENSIX_KERNELS`, else off) |
| `run EXPORT PROMPT` | `--raw`, `--think`, `--trace` (all off), `--max-new 256`, `--max-seq 2048`, `--trace-mb 64`, `--device-id 0`, `--kernels PATH` |
| `chat EXPORT` | `--system TEXT` (none), `--think`, `--trace` (off), `--max-new 256`, `--max-seq 2048`, `--trace-mb 64`, `--device-id 0`, `--kernels PATH` |
| `bench EXPORT` | `--pp 512`, `--tg 128`, `--depth 0`, `--repeat 3`, `--max-seq 4096`, `--trace` (off), `--trace-mb 64`, `--device-id 0`, `--kernels PATH` |

For `pcc`, `--greedy`, `--spec`, and `--window` are nonnegative. `--verify` is zero or 2–32
(2–8 with kernels); `--mtp` requires `--prefill` 2–32 (2–8 with kernels). For `bench`,
`--pp` and `--repeat` are positive, `--tg` is positive, and `--depth` is nonnegative; `--pp`
and `--depth` accept comma-separated values. `--kernels` and `--trace` have the same defaults
across the commands that accept them.

## Tests

```
pytest                               # host tests; the device tests need ttnn and a card or ttsim
TENSIX_RUNNER_EXPORT=exports/Qwen3-4B-Tenstorrent-Tensix-BFP4 TENSIX_RUNNER_LAYERS=all pytest -m export
```

`TENSIX_RUNNER_LAYERS` selects comma-separated layers for the real-export device test, or `all`;
by default it tests the first and last manifest layers.

On ttsim the suite sets `TT_METAL_DISABLE_SFPLOADMACRO=1` (unless already set), because the
simulator does not support that instruction.

## License

Apache 2.0 (`LICENSE`, `NOTICE`). Files adapted from other projects are listed in `ORIGINS.md`.
Each export carries its base model's license.
