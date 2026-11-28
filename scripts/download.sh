#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Xylolabs
#
# download.sh NAME [--repo ORG/NAME] [--dir DIR] [--variant FOLDER]
# Fetch into exports/<repo name> (or DIR); folder exports go into their variant folder beneath it.
# The repo defaults to Xylolabs/NAME. PYTHON selects the interpreter (default: .venv/bin/python).
set -euo pipefail
cd "$(dirname "$0")/.."
py=${PYTHON:-.venv/bin/python}
exec "$py" -m tensix_runner download "$@"
