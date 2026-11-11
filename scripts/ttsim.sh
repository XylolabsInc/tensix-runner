# shellcheck shell=bash
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Xylolabs
#
# source scripts/ttsim.sh bh|wh [DIR]: point ttnn at Tenstorrent's simulator instead of a card. DIR holds
# libttsim_<arch>.so and its soc_descriptor.yaml (default: ./ttsim/<arch>, as scripts/install.sh --ttsim puts them).
# ttsim runs slow dispatch only, so traces do not run on it, and it does not model the SFPLOADMACRO instruction.
_arch=${1:-bh}
_dir=${2:-$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." && pwd)/ttsim/$_arch}
if [ ! -f "$_dir/libttsim_$_arch.so" ] || [ ! -f "$_dir/soc_descriptor.yaml" ]; then
	echo "ttsim.sh: no libttsim_$_arch.so and soc_descriptor.yaml in $_dir (scripts/install.sh --ttsim $_arch)" >&2
	return 1 2>/dev/null || exit 1
fi
export TT_METAL_SIMULATOR=$_dir/libttsim_$_arch.so
export TT_METAL_SLOW_DISPATCH_MODE=1
export TT_METAL_DISABLE_SFPLOADMACRO=1
unset _arch _dir
