<div align="center">

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/XylolabsInc/tensix-bfp-gptq/main/assets/xylolabs-lockup-dark.svg">
  <img src="https://raw.githubusercontent.com/XylolabsInc/tensix-bfp-gptq/main/assets/xylolabs-lockup.svg" alt="Xylolabs" height="40">
</picture>

# tensix-runner

**Run Xylolabs' BFP4 and BFP8 Hugging Face exports on Tenstorrent Tensix devices, through upstream ttnn**

[![License](https://img.shields.io/badge/license-Apache--2.0-blue)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.12-3776AB)](pyproject.toml)
[![ttnn](https://img.shields.io/badge/ttnn-0.81.1-5B2C83)](https://github.com/tenstorrent/tt-metal/releases/tag/v0.81.1)
[![Hugging Face](https://img.shields.io/badge/Hugging%20Face-models-FFD21E)](https://huggingface.co/Xylolabs)

`tenstorrent` · `tensix` · `tt-metal` · `ttnn` · `wormhole` · `blackhole` · `llm-inference` · `block-floating-point`

</div>

Runs Xylolabs' Hugging Face exports of LLMs, quantized with GPTQ to TT-NN's block floating point
formats `ttnn.bfloat4_b` (BFP4) and `ttnn.bfloat8_b` (BFP8) by
[tensix-bfp-gptq](https://github.com/XylolabsInc/tensix-bfp-gptq), on a Tenstorrent Wormhole or
Blackhole chip through the public ttnn API of [tt-metal](https://github.com/tenstorrent/tt-metal)
v0.81.1. The embedding lookup, the LM head and its argmax, and the drafter of models with multi-token
prediction run on the device; the host tokenizes and schedules. Not affiliated with or endorsed by
Tenstorrent.

[DEVELOPMENT.md](DEVELOPMENT.md) says how this repository is developed, from Tenstorrent's open software stack only; [ORIGINS.md](ORIGINS.md) says where every file comes from and how that was checked.

## Models

| Family | Exports | Chips |
|---|---|---|
| Qwen3 | Qwen3-4B, Qwen3-8B | Blackhole, Wormhole |
| Qwen3.5 (with MTP drafting) | Qwen3.5-4B, Qwen3.5-9B | Blackhole |
| Gemma 4 | gemma-4-E2B, gemma-4-E4B | Blackhole, Wormhole |

Tested on [ttsim](https://github.com/tenstorrent/ttsim) v1.16.0, Tenstorrent's simulator, not yet on
a card. `--trace` (decode replayed from a trace) needs a card and has not run yet.

## Install

Linux x86_64 and Python 3.12: ttnn 0.81.1 ships wheels for CPython 3.12 (and 3.10) on Linux x86_64
only, so newer Python versions wait on Tenstorrent's wheels.

```
git clone https://github.com/XylolabsInc/tensix-runner.git && cd tensix-runner
scripts/install.sh                 # .venv with tensix-runner, ttnn 0.81.1, torch (CPU) and SFPI 7.89.0
source .venv/bin/activate
```

`install.sh` uses [uv](https://docs.astral.sh/uv/) when it is installed. Without uv it runs
`python3.12 -m venv`, so it needs a Python 3.12 whose `venv` module works (on Debian and Ubuntu,
the `python3.12-venv` package).

`install.sh` unpacks SFPI 7.89.0, tt-metal v0.81.1's compiler for the Tensix cores, into ttnn's
`runtime` directory, because the wheel does not include it and `python -m ttnn.download_sfpi` does
not work in 0.81.1. A card also needs Tenstorrent's driver and firmware
([tt-installer](https://github.com/tenstorrent/tt-installer)).

**Without a card,** ttsim stands in for one:

```
scripts/install.sh --ttsim bh      # or wh: libttsim and its SoC descriptor under ttsim/<arch>/
source scripts/ttsim.sh bh         # TT_METAL_SIMULATOR, slow dispatch
tensix-runner info                 # e.g. "blackhole (simulated): compute grid ..."
```

ttsim runs slow dispatch only, so traced decoding does not run on it, and it is much slower than a
card.

## Get an export

```
tensix-runner download Qwen3-4B-Tenstorrent-Tensix-BFP4  # from Xylolabs/Qwen3-4B-Tenstorrent-Tensix-BFP4 into exports/
tensix-runner download NAME --repo ORG/NAME --dir DIR
```

Each export's model card lists its formats per tensor class, its size and its accuracy against the
bf16 model. `quant4.json` in the export names each projection's format; the runner reads it.

## Run

```
tensix-runner run exports/Qwen3-4B-Tenstorrent-Tensix-BFP4 "What is the capital of France?"
tensix-runner chat exports/Qwen3-4B-Tenstorrent-Tensix-BFP4 --system "You are concise."
tensix-runner bench exports/Qwen3-4B-Tenstorrent-Tensix-BFP4 --pp 512 --tg 128 --depth 0
```

`run` and `chat` render the checkpoint's own chat template (`--think` turns on thinking where the
template has it) and decode greedily, with the argmax on the device. `bench` times prefill (`pp`)
and decode (`tg`) in llama-bench's terms on the device you run it on. `--trace` on any of them
replays the decode step from a trace (a card only).

## Kernels (optional)

```
tensix-runner run exports/gemma-4-E2B-it-Tenstorrent-Tensix-BFP4-MLP "Hello" --kernels ../tensix-kernels
export TENSIX_KERNELS=../tensix-kernels      # the same for every command
```

With a [tensix-kernels](https://github.com/XylolabsInc/tensix-kernels) checkout, every matrix
runs through its dense reader and the LM head through its device head, which picks the token on the
device. `--trace` does not take the kernels' head yet.

## Compare with transformers

```
pip install -e ".[ref]"
tensix-runner pcc exports/Qwen3-4B-Tenstorrent-Tensix-BFP4 --layers 0,17,35 --prefill 32 --decode 2
tensix-runner pcc exports/Qwen3-4B-Tenstorrent-Tensix-BFP4 --greedy 8
tensix-runner pcc exports/gemma-4-E2B-it-Tenstorrent-Tensix-BFP4-MLP --drift
```

`--layers` runs each listed layer on transformers' own inputs and prints its PCC and largest
difference (a layer that reads an earlier layer's KV cache brings that layer along). `--greedy`
has the device decode transformers' greedy tokens and compares every step's token; a difference
counts as a tie when transformers' two logits lie within one bfloat16 step. `--drift` compares
every layer's output within one whole-model prefill, where errors add up. `--window 32` sets
Gemma 4's sliding window on both sides, so that a short prompt reaches the window's edge.
transformers runs the export in float32 on the CPU, which needs about four times the export's bf16
size in memory.

## Check an export

```
tensix-runner check exports/Qwen3-4B-Tenstorrent-Tensix-BFP4 --host     # torch alone: every matrix on its grid
tensix-runner check exports/Qwen3-4B-Tenstorrent-Tensix-BFP4            # through ttnn's packer
tensix-runner check exports/Qwen3-4B-Tenstorrent-Tensix-BFP4 --device   # and through device DRAM and back
```

Each matrix is transposed to `[in, out]`, the layout TT-NN's matmul reads, packed at its format and
unpacked again; the check passes when every value comes back bit for bit. `--layers 0,1` limits it.

## Tests

```
pytest                               # host tests; the device tests need ttnn and a card or ttsim
TENSIX_RUNNER_EXPORT=exports/Qwen3-4B-Tenstorrent-Tensix-BFP4 pytest -m export
```

## License

Apache 2.0 (`LICENSE`, `NOTICE`). Files adapted from other projects are listed in `ORIGINS.md`.
Each export carries its base model's license.
