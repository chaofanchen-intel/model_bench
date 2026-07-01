#!/bin/bash
# Build the model_bench C++ suite.
#
#   ./build.sh              # Release build into ./build
#   ./build.sh --opencv     # also build VLM --image support (needs OpenCV dev pkgs)
#   ./build.sh --clean      # wipe ./build first
set -eo pipefail
# Use an uncommon name: setupvars.sh leaks its own SCRIPT_DIR into our shell.
MB_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$MB_ROOT"

# setupvars.sh references unset vars; source it with `set -u` off.
# shellcheck disable=SC1091
source /opt/intel/openvino_2026/setupvars.sh
# shellcheck disable=SC1091
source /opt/intel/openvino_genai/setupvars.sh

CMAKE_ARGS=(-DCMAKE_BUILD_TYPE=Release)
for arg in "$@"; do
    case "$arg" in
        --opencv) CMAKE_ARGS+=(-DMB_WITH_OPENCV=ON) ;;
        --clean) rm -rf build ;;
        *) echo "unknown arg: $arg"; exit 1 ;;
    esac
done

cmake -S "$MB_ROOT" -B "$MB_ROOT/build" "${CMAKE_ARGS[@]}"
cmake --build "$MB_ROOT/build" -j"$(nproc)"
echo ""
echo "Built: $MB_ROOT/build/model_bench"
