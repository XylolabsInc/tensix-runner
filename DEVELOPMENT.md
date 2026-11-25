# How this repository is developed

tensix-runner is written for Tenstorrent Tensix NPUs from Tenstorrent's open software stack only.

## What is used

- **Upstream code and tools:** tt-metal and TT-NN at the version pinned in this repository (v0.81.1), the ttsim
  simulator, tt-forge and tt-mlir where named, and Tenstorrent's public documentation and community answers.
- **Model code and weights:** public Hugging Face repositories under the licenses their cards declare; files adapted
  from them keep their copyright lines and are listed in ORIGINS.md.
- **Our own designs:** each change starts as a written specification of our own design (the algorithm, the data
  layout, the API, the tests and their tolerances). The specification is reviewed before any code is written, and it
  contains no third-party code.

## What is not used

While code for this repository is developed, nothing is used from any hardware vendor other than Tenstorrent's open
releases: no other vendor's SDK, source code, documentation, test facility or hardware. Defaults that depend on a
device (core counts, memory banks, L1 sizes) are read from the device at run time; tuned settings are starting
points, not measurements of a particular device.

## How a change is made

1. A specification is written and reviewed for third-party content.
2. A developer who works only from the specification and the public stack implements it in a clean checkout that
   holds this repository, the pinned upstream release and ttsim, and nothing else.
3. The change is tested on ttsim with the checks named in AGENTS.md and RUNNING.md; correctness is checked against
   float64 or bf16 references with the tolerances the specification fixes.
4. Every file's origin is recorded in ORIGINS.md (and NOTICE for files that follow tt-metal), with SPDX headers.
5. Each commit is reviewed for third-party content, including a token comparison against code we may not use, before
   it is pushed; commits are signed.

## Releases before this process

v0.1.0 (2026-11-16) predates it. It was written by Xylolabs from upstream tt-metal, TT-NN and ttsim only (see
ORIGINS.md), and before release it was checked by token comparison against code we may not use, with no match beyond
upstream code. Releases after v0.1.0 follow the process above.

## Log

| Date | Change | Specification (name, MD5) | Upstream | Checks |
|---|---|---|---|---|
| 2026-11-24 | docs(origins): record every file's origin | 01-origins-md.md, 0fb963254532e4ff23a9568d4dce868a | tt-metal/TT-NN v0.81.1, ttsim v1.16.0 | pytest on ttsim (41 passed, 9 skipped; the 3 test_window_decode cases end the simulator process, as they do on the commit before this change, and were left out), tests/test_origins.py (2 passed), ruff check and format |
| 2026-11-25 | fix(export): find the LM head by the name quant4.json declares | 01-lm-head-tensor-name.md, 4945255f42a11f2eb6cd18adfd8ac88c | tt-metal/TT-NN v0.81.1, ttsim v1.16.0 | pytest on ttsim (58 passed, 9 skipped, including tests/test_origins.py; the 3 test_window_decode cases end the simulator process, as they do on main, and were left out), ruff check and format, shellcheck |
| 2026-11-25 | fix(export): find the LM head by the name quant4.json declares | 01-lm-head-tensor-name.md, 4945255f42a11f2eb6cd18adfd8ac88c | tt-metal/TT-NN v0.81.1, ttsim v1.16.0 | pytest on ttsim (60 passed, 9 skipped, including tests/test_origins.py; the 3 test_window_decode cases end the simulator process, as they do on main, and were left out), ruff check and format, shellcheck |
| 2026-11-25 | fix(cli): check the LM head too | 02-check-covers-head.md, 67291faf28361bf7eb41b766597f6945 | tt-metal/TT-NN v0.81.1, ttsim v1.16.0 | pytest on ttsim (67 passed, 9 skipped, including tests/test_origins.py; the 3 test_window_decode cases end the simulator process, as they do on main, and were left out), ruff check and format, shellcheck |
| 2026-11-25 | fix(cli): check counts and lengths before the runner runs | 03-bounds-and-argument-checks.md, 7e8d82787089aed03008f01ccca74b66 | tt-metal/TT-NN v0.81.1, ttsim v1.16.0 | pytest on ttsim (97 passed, 9 skipped, including tests/test_origins.py and 30 new tests in tests/test_bounds.py; the 3 test_window_decode cases end the simulator process, as they do on main, and were left out), ruff check and format, shellcheck |
| 2026-11-25 | test(conftest): ✅ run the suite on ttsim without the unsupported instruction | 05-ttsim-sfploadmacro-env.md, c2260cbe2092c5123d3089a29ca2deff | tt-metal/TT-NN v0.81.1, ttsim v1.16.0 | pytest on ttsim with nothing deselected (103 passed, 9 skipped, exit 0, including the 3 test_window_decode cases, tests/test_origins.py and 3 new tests in tests/test_conftest_env.py), ruff format; ruff check reports only the E402 in tests/test_ops.py that main already has |
