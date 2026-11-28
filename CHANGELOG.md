# Changelog

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
