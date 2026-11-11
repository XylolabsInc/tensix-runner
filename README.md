<div align="center">

# tensix-runner

**Run Xylolabs' BFP4 and BFP8 Hugging Face exports on Tenstorrent Tensix devices, through upstream ttnn**

[![License](https://img.shields.io/badge/license-Apache--2.0-blue)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.12-3776AB)](pyproject.toml)
[![ttnn](https://img.shields.io/badge/ttnn-0.81.1-5B2C83)](https://github.com/tenstorrent/tt-metal/releases/tag/v0.81.1)
[![Hugging Face](https://img.shields.io/badge/Hugging%20Face-models-FFD21E)](https://huggingface.co/Xylolabs)

`tenstorrent` · `tensix` · `tt-metal` · `ttnn` · `wormhole` · `blackhole` · `llm-inference` · `block-floating-point`

</div>

Xylolabs publishes LLM checkpoints whose linear layers already sit on TT-NN's block floating point
grids, `ttnn.bfloat4_b` (BFP4) and `ttnn.bfloat8_b` (BFP8), rounded by GPTQ with
[tensix-bfp-gptq](https://github.com/XylolabsInc/tensix-bfp-gptq). This runner loads them onto a
Wormhole or Blackhole chip with every value packed exactly as stored, and runs them with the work on
the device: the embedding lookup, the LM head and its argmax, and the drafter of models with
multi-token prediction. The host only tokenizes and schedules.

It is written against the public ttnn API of [tt-metal](https://github.com/tenstorrent/tt-metal)
v0.81.1 and needs only the `ttnn` wheel. Not affiliated with or endorsed by Tenstorrent.

## Where it stands

| Part | State |
|---|---|
| Loading an export, every matrix packed exactly | Blackhole and Wormhole |
| Qwen3 dense (4B, 8B) | Blackhole and Wormhole: layers match transformers (below) |
| Qwen3.5 (4B, 9B: Gated DeltaNet, gated attention) | Blackhole: layers match transformers; Wormhole not yet (ttnn's Gated Delta Rule ops stop on ttsim's Wormhole) |
| Qwen3.5's MTP drafting on the device | Blackhole: drafts and verifies; the tokens equal plain greedy decoding on a 4-layer check, whose drafts the model never accepts, so acceptance is not measured yet |
| Gemma 4 (E2B, E4B: per-layer inputs, sliding and full attention, KV sharing across layers) | Blackhole and Wormhole: layers match transformers (below) |
| `run`, `chat`, `bench` | work; decode is eager by default, `--trace` replays it from a trace on a card |

Nothing here has run on a Tenstorrent card yet. Every result comes from
[ttsim](https://github.com/tenstorrent/ttsim) v1.16.0, Tenstorrent's functional simulator, and this
README gives no speeds. ttsim has no traces, so `--trace` has not run anywhere yet.

- **Exact weights.** On both of its chips, the Blackhole (11 x 10 compute grid, 8 DRAM banks) and
  the Wormhole (8 x 8, 12 banks), all 252 matrices of
  [Qwen3-4B-Tensix-BFP4](https://huggingface.co/Xylolabs/Qwen3-4B-Tensix-BFP4) came back from
  device DRAM bit for bit (`tensix-runner check --device`).
- **Layers against transformers** (`tensix-runner pcc`): each layer gets transformers' own inputs
  (float32, CPU, the same export) for a 32-token prefill and then two decode steps, and every
  output reaches a Pearson correlation of 0.9997 or more with transformers' output: Qwen3-4B's
  layers 0, 17 and 35, Qwen3-8B's layers 0, 18 and 35, Gemma 4 E2B's layers 0, 4, 13, 14, 15 and
  19 and E4B's layers 0, 5, 22, 23, 24 and 29 (sliding, full and KV-sharing layers) on both chips,
  and Qwen3.5-4B's layers 0, 3, 16, 30 and 31 (DeltaNet and gated attention) on the Blackhole.
- **Greedy steps against transformers** (`--greedy 8`, the device decoding transformers' tokens
  after a 32-token prompt): every step's token equal for Qwen3-4B and Gemma 4 E2B on both chips
  and for Qwen3.5-4B and Gemma 4 E4B on the Blackhole.
- **Through the whole depth** (`--drift`): in one prefill on the Blackhole, every one of Gemma 4
  E2B's 35 layer outputs stays at PCC 0.99986 or more against transformers' float32 run, about
  where transformers' own bf16 run stays (0.99987).

## Install

Linux x86_64 and Python 3.12: ttnn 0.81.1 ships wheels for CPython 3.12 (and 3.10) on Linux x86_64
only, so newer Python versions wait on Tenstorrent's wheels.

```
git clone https://github.com/XylolabsInc/tensix-runner.git && cd tensix-runner
scripts/install.sh                 # .venv with tensix-runner, ttnn 0.81.1, torch (CPU) and SFPI 7.89.0
source .venv/bin/activate
```

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
tensix-runner download Qwen3-4B-Tensix-BFP4          # from Xylolabs/Qwen3-4B-Tensix-BFP4 into exports/
tensix-runner download NAME --repo ORG/NAME --dir DIR
```

Each export's model card lists its formats per tensor class, its size and its accuracy against the
bf16 model. `quant4.json` in the export names each projection's format; the runner reads it.

## Run

```
tensix-runner run exports/Qwen3-4B-Tensix-BFP4 "What is the capital of France?"
tensix-runner chat exports/Qwen3-4B-Tensix-BFP4 --system "You are concise."
tensix-runner bench exports/Qwen3-4B-Tensix-BFP4 --pp 512 --tg 128 --depth 0
```

`run` and `chat` render the checkpoint's own chat template (`--think` turns on thinking where the
template has it) and decode greedily, with the argmax on the device. `bench` times prefill (`pp`)
and decode (`tg`) in llama-bench's terms on the device you run it on. `--trace` on any of them
replays the decode step from a trace (a card only).

## Compare with transformers

```
pip install -e ".[ref]"
tensix-runner pcc exports/Qwen3-4B-Tensix-BFP4 --layers 0,17,35 --prefill 32 --decode 2
tensix-runner pcc exports/Qwen3-4B-Tensix-BFP4 --greedy 8
tensix-runner pcc exports/gemma-4-E2B-it-Tensix-BFP4-MLP --drift
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
tensix-runner check exports/Qwen3-4B-Tensix-BFP4 --host     # torch alone: every matrix on its grid
tensix-runner check exports/Qwen3-4B-Tensix-BFP4            # through ttnn's packer
tensix-runner check exports/Qwen3-4B-Tensix-BFP4 --device   # and through device DRAM and back
```

Each matrix is transposed to `[in, out]`, the layout TT-NN's matmul reads, packed at its format and
unpacked again; the check passes when every value comes back bit for bit. `--layers 0,1` limits it.

## Tests

```
pytest                               # host tests; the device tests need ttnn and a card or ttsim
TENSIX_RUNNER_EXPORT=exports/Qwen3-4B-Tensix-BFP4 pytest -m export
```

## License

Apache 2.0 (`LICENSE`, `NOTICE`). Files adapted from other projects are listed in `ORIGINS.md`.
Each export carries its base model's license.
