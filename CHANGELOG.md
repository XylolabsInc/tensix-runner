# Changelog

## 0.1.3
### Fixed
- The greedy reference steps transformers' model on its KV cache and takes each token from its own logits row, so an
  end-of-sequence token that is the raw argmax is no longer masked; a tie needs a finite, non-negative gap.
- `pcc` fails a layer, drift or MTP check whose PCC is NaN or infinite, and refuses a non-finite `--bar`.
- The `pcc` prompt text is encoded without special tokens, so a BOS post-processor adds no second BOS.
- `run` and `chat` print a replacement character held back at the end of the output and keep it in the chat history.
- `speculate` with one new token needs only the prompt to fit in `max_seq`.
- The matmul accumulation mode follows the device that runs the matmul, not the last model loaded.

## 0.1.2
### Fixed
- Qwen3.5 prefill normalizes Q and K per head, as decode and verify do.
- Speculation commits only the rows it returns, reports the kept drafts correctly and drafts nothing for a request
  that prefill already completed.
- Decode rows are sharded over the device's compute grid, not a single row of cores.
- `pcc` refuses combinations of primary checks, sizes its cache without an unused `--k`; `bench` warms every prompt
  length and a decode step before timing; the token count must be a non-negative integer.
### Changed
- Streamed text is decoded incrementally; compute configs are built once.
