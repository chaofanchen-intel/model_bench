#!/bin/bash
# Common helpers for the model_bench C++ runner scripts.
# These mirror tests/runners/_common.sh but invoke the compiled model_bench
# binary instead of Python. Source the OpenVINO + GenAI environments and locate
# the binary.

# Set up the OpenVINO runtime + GenAI environments (idempotent).
# setupvars.sh references unset vars, so disable `set -u`/`-e` while sourcing.
mb_setup_env() {
    local had_u had_e
    case "$-" in *u*) had_u=1 ;; *) had_u=0 ;; esac
    case "$-" in *e*) had_e=1 ;; *) had_e=0 ;; esac
    set +ue
    # Prefer the locally-built OpenVINO (the one model_bench/build-local links against)
    # when present; fall back to a system /opt/intel install otherwise.
    local local_ov="${MB_DIR}/../benchmark/openvino/install"
    if [ -f "$local_ov/setupvars.sh" ]; then
        # shellcheck disable=SC1091
        source "$local_ov/setupvars.sh" >/dev/null 2>&1
        export LD_LIBRARY_PATH="$local_ov/runtime/lib/intel64:${LD_LIBRARY_PATH:-}"
    else
        # shellcheck disable=SC1091
        [ -f /opt/intel/openvino_2026/setupvars.sh ] && source /opt/intel/openvino_2026/setupvars.sh >/dev/null 2>&1
        # shellcheck disable=SC1091
        [ -f /opt/intel/openvino_genai/setupvars.sh ] && source /opt/intel/openvino_genai/setupvars.sh >/dev/null 2>&1
    fi
    [ "$had_u" = 1 ] && set -u
    [ "$had_e" = 1 ] && set -e
    return 0
}

# Locate the model_bench binary (BIN env overrides).
mb_select_bin() {
    if [ -n "${BIN:-}" ]; then
        MB_BIN="$BIN"
    elif [ -x "${MB_DIR}/build-local/model_bench" ]; then
        # build-local links against the locally-built OpenVINO (see build.sh / docs).
        MB_BIN="${MB_DIR}/build-local/model_bench"
    elif [ -x "${MB_DIR}/build/model_bench" ]; then
        MB_BIN="${MB_DIR}/build/model_bench"
    else
        echo "[ERROR] model_bench binary not found; build it first:" >&2
        echo "        cd ${MB_DIR} && cmake -S . -B build-local -DCMAKE_BUILD_TYPE=Release && cmake --build build-local -j" >&2
        exit 1
    fi
}

# Parse devices: positional args > $DEVICE > "CPU". Sets DEVICES array.
mb_parse_devices() {
    if [ $# -gt 0 ]; then
        DEVICES=("$@")
    elif [ -n "${DEVICE:-}" ]; then
        read -ra DEVICES <<< "$DEVICE"
    else
        DEVICES=("CPU")
    fi
}

# Pre-authenticate sudo for GPU/NPU telemetry when monitoring is on.
mb_refresh_sudo() {
    [ "${MONITOR:-0}" = "1" ] || return 0
    command -v sudo >/dev/null 2>&1 || { echo "[WARN] sudo not found; GPU/NPU util may be unavailable."; return 0; }
    echo "[INFO] Refreshing sudo for xpu-smi/NPU telemetry..."
    sudo -v || echo "[WARN] sudo auth failed; GPU/NPU util may be unavailable."
}
