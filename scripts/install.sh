#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Xylolabs
#
# install.sh [--ttsim bh|wh] [--venv DIR]: a Python 3.12 virtual environment with tensix-runner, ttnn 0.81.1
# (Linux x86_64 only) and torch for the CPU, then SFPI 7.89.0 unpacked into ttnn's runtime directory (the ttnn
# wheel does not ship it, and `python -m ttnn.download_sfpi` does not work in 0.81.1). With --ttsim, also
# Tenstorrent's simulator for Blackhole (bh) or Wormhole (wh) under ./ttsim/<arch>/, with the SoC descriptor it
# reads beside it. Then `source scripts/ttsim.sh bh` (or wh) before running anything on the simulator.
set -euo pipefail
cd "$(dirname "$0")/.."
TTSIM_VERSION=v1.16.0
SFPI_VERSION=7.89.0
TT_METAL_VERSION=v0.81.1
venv=.venv
sim=
while [ $# -gt 0 ]; do
	case $1 in
	--ttsim) sim=${2:?--ttsim bh or wh}; shift 2 ;;
	--venv) venv=${2:?--venv DIR}; shift 2 ;;
	*) echo "usage: install.sh [--ttsim bh|wh] [--venv DIR]" >&2; exit 2 ;;
	esac
done
case $sim in '' | bh | wh) ;; *) echo "install.sh: --ttsim takes bh or wh" >&2; exit 2 ;; esac
if [ "$(uname -s)-$(uname -m)" != Linux-x86_64 ]; then
	echo "install.sh: ttnn 0.81.1 ships wheels for Linux x86_64 only" >&2
	exit 1
fi
if command -v uv >/dev/null; then
	uv venv --python 3.12 "$venv"
	pip=(uv pip install --python "$venv/bin/python")
else
	python3.12 -m venv "$venv"
	pip=("$venv/bin/python" -m pip install)
fi
# torch for the CPU first, from its own index only (so no other package resolves there), then the rest from PyPI
"${pip[@]}" torch --index-url https://download.pytorch.org/whl/cpu
"${pip[@]}" -e ".[ttnn,ref,test]"
rt=$("$venv/bin/python" -c 'import os, ttnn; print(os.path.join(os.path.dirname(ttnn.__file__), "runtime"))')
if [ ! -d "$rt/sfpi" ]; then
	# the wheel names its SFPI release and the archive's sha256 (tt_metal/sfpi-version, NAME='VALUE' lines), read without running it
	sv=$(dirname "$rt")/tt_metal/sfpi-version
	sfpi_version=$(sed -n "s/^sfpi_version='\\([^']*\\)'\$/\\1/p" "$sv")
	sfpi_x86_64_debian_txz_hash=$(sed -n "s/^sfpi_x86_64_debian_txz_hash='\\([0-9a-f]*\\)'\$/\\1/p" "$sv")
	[ "$sfpi_version" = "$SFPI_VERSION" ] || { echo "the ttnn wheel names SFPI ${sfpi_version:-?}, not $SFPI_VERSION" >&2; exit 1; }
	tmp=$(mktemp -d)
	curl -fsSL -o "$tmp/sfpi.txz" \
		"https://github.com/tenstorrent/sfpi/releases/download/$SFPI_VERSION/sfpi_${SFPI_VERSION}_x86_64_debian.txz"
	echo "${sfpi_x86_64_debian_txz_hash:?the wheel names no SFPI hash}  $tmp/sfpi.txz" | sha256sum -c --quiet || exit 1
	tar -xJf "$tmp/sfpi.txz" -C "$rt"
	rm -rf "$tmp"
fi
if [ -n "$sim" ]; then
	soc=$([ "$sim" = bh ] && echo blackhole_140_arch.yaml || echo wormhole_b0_80_arch.yaml)
	mkdir -p "ttsim/$sim"
	curl -fsSL -o "ttsim/$sim/libttsim_$sim.so" \
		"https://github.com/tenstorrent/ttsim/releases/download/$TTSIM_VERSION/libttsim_$sim.so"
	# ttsim reads the chip's layout from soc_descriptor.yaml beside the library (tt-metal before v0.83)
	curl -fsSL -o "ttsim/$sim/soc_descriptor.yaml" \
		"https://raw.githubusercontent.com/tenstorrent/tt-metal/$TT_METAL_VERSION/tt_metal/soc_descriptors/$soc"
fi
echo "done: source $venv/bin/activate${sim:+ && source scripts/ttsim.sh $sim}"
