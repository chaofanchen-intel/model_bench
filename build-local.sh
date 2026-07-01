#!/bin/bash
# Build the model_bench C++ suite.
#
#   ./build-local.sh              # Release build into ./build-local
#   ./build-local.sh --opencv     # also build VLM --image support (needs OpenCV dev pkgs)
#   ./build-local.sh --clean      # wipe ./build-local first
set -eo pipefail
# Use an uncommon name: setupvars.sh leaks its own SCRIPT_DIR into our shell.
MB_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$MB_ROOT"

# Point OV_SETUPVARS at your local OpenVINO install's setupvars.sh, or set
# OV_DIR to the OpenVINO install root. Defaults assume an install tree next to
# this repo (../benchmark/openvino/install).
OV_SETUPVARS="${OV_SETUPVARS:-${OV_DIR:-$MB_ROOT/../benchmark/openvino/install}/setupvars.sh}"
if [ ! -f "$OV_SETUPVARS" ]; then
    echo "OpenVINO setupvars.sh not found at: $OV_SETUPVARS" >&2
    echo "Set OV_SETUPVARS=/path/to/openvino/install/setupvars.sh (or OV_DIR) and retry." >&2
    exit 1
fi
# setupvars.sh references unset vars; source it with `set -u` off.
# shellcheck disable=SC1091
source "$OV_SETUPVARS"

CMAKE_ARGS=(-DCMAKE_BUILD_TYPE=Release)
for arg in "$@"; do
    case "$arg" in
        --opencv) CMAKE_ARGS+=(-DMB_WITH_OPENCV=ON) ;;
        --clean) rm -rf build-local ;;
        *) echo "unknown arg: $arg"; exit 1 ;;
    esac
done

cmake -S "$MB_ROOT" -B "$MB_ROOT/build-local" "${CMAKE_ARGS[@]}"
cmake --build "$MB_ROOT/build-local" -j"$(nproc)"
echo ""
echo "Built: $MB_ROOT/build-local/model_bench"
