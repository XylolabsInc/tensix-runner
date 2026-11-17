#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Xylolabs
#
# download.sh NAME [--repo ORG/NAME] [--dir DIR]: fetch an export from Hugging Face into exports/NAME (or DIR),
# e.g. download.sh Qwen3-4B-Tenstorrent-Tensix-BFP4. The repo defaults to Xylolabs/NAME.
set -euo pipefail
cd "$(dirname "$0")/.."
py=${PYTHON:-.venv/bin/python}
exec "$py" -m tensix_runner download "$@"
