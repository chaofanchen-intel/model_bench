#!/bin/bash
#
# model_bench LLMLingua-2 runner (raw ov::Core; supports --profile).
#
# Usage:
#   ./runners/run_lingua.sh CPU GPU NPU
#   MODELS="../models/llmlingua-2-xlm-roberta-large-meetingbank-INT8" PROFILE=1 ./runners/run_lingua.sh CPU
#
# Env overrides (defaults in []):
#   MODELS         [../models/llmlingua-2-bert-base-multilingual-cased-meetingbank-INT8]
#   DEVICE         [CPU]
#   INPUT_LENGTHS  [64 128 256 512]
#   RATES          [0.3 0.5 0.7]
#   WARMUP         [5]
#   ITERS          [50]
#   MONITOR        [0]
#   PROFILE        [0]   1 -> per-op ProfilingInfo hotspot report (the C++ payoff)
#   PROFILE_CSV    [results/lingua_profile.csv when PROFILE=1]
#   CSV / JSON     [results/lingua.{csv,json}]
#   EXTRA          []
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MB_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
source "$SCRIPT_DIR/_common.sh"
mb_setup_env
mb_select_bin
mb_parse_devices "$@"

MODELS="${MODELS:-$MB_DIR/../models/llmlingua-2-bert-base-multilingual-cased-meetingbank-INT8}"
INPUT_LENGTHS="${INPUT_LENGTHS:-64 128 256 512}"
RATES="${RATES:-0.3 0.5 0.7}"
WARMUP="${WARMUP:-5}"
ITERS="${ITERS:-50}"
CSV="${CSV:-$MB_DIR/results/lingua.csv}"
JSON="${JSON:-$MB_DIR/results/lingua.json}"

read -ra LEN_ARR <<< "$INPUT_LENGTHS"
read -ra RATE_ARR <<< "$RATES"
ARGS=(-d "${DEVICES[@]}" --input-lengths "${LEN_ARR[@]}" --compression-rates "${RATE_ARR[@]}"
      --warmup "$WARMUP" --iters "$ITERS" -r "$CSV" -rj "$JSON")
[ "${MONITOR:-0}" = "1" ] && ARGS+=(--monitor)
if [ "${PROFILE:-0}" = "1" ]; then
    ARGS+=(--profile --profile-csv "${PROFILE_CSV:-$MB_DIR/results/lingua_profile.csv}")
fi
[ -n "${EXTRA:-}" ] && read -ra EXTRA_ARR <<< "$EXTRA" && ARGS+=("${EXTRA_ARR[@]}")

mb_refresh_sudo
read -ra MODEL_ARR <<< "$MODELS"
echo "[RUN] model_bench lingua -m ${MODEL_ARR[*]} ${ARGS[*]}"
"$MB_BIN" lingua -m "${MODEL_ARR[@]}" "${ARGS[@]}"
