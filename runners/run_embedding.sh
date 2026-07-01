#!/bin/bash
#
# model_bench Qwen3-Embedding runner.
#
# Usage:
#   ./runners/run_embedding.sh CPU GPU NPU
#   MODELS="../models/Qwen3-Embedding-0.6B-int8-ov" MAX_LENGTH=512 ITERS=100 ./runners/run_embedding.sh GPU
#
# Env overrides (defaults in []):
#   MODELS        [../models/Qwen3-Embedding-0.6B-int8-ov]
#   DEVICE        [CPU]
#   ITERS         [5]
#   MAX_LENGTH    []                 --max-length (Qwen3 encoders default large)
#   BATCH_SIZE    [1]
#   MODE          [documents]        documents | query | both
#   POOLING       [last_token]
#   PROFILE       [0]                1 -> raw-Core per-op graph profiling
#   PROFILE_CSV   [results/embedding_profile.csv when PROFILE=1]
#   MONITOR       [0]
#   CSV / JSON    [results/embedding.{csv,json}]
#   EXTRA         []
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MB_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
source "$SCRIPT_DIR/_common.sh"
mb_setup_env
mb_select_bin
mb_parse_devices "$@"

MODELS="${MODELS:-$MB_DIR/../models/Qwen3-Embedding-0.6B-int8-ov}"
ITERS="${ITERS:-5}"
BATCH_SIZE="${BATCH_SIZE:-1}"
MODE="${MODE:-documents}"
POOLING="${POOLING:-last_token}"
CSV="${CSV:-$MB_DIR/results/embedding.csv}"
JSON="${JSON:-$MB_DIR/results/embedding.json}"

ARGS=(-d "${DEVICES[@]}" -n "$ITERS" -bs "$BATCH_SIZE" --mode "$MODE" --pooling-type "$POOLING"
      -r "$CSV" -rj "$JSON")
[ -n "${MAX_LENGTH:-}" ] && ARGS+=(--max-length "$MAX_LENGTH")
[ "${MONITOR:-0}" = "1" ] && ARGS+=(--monitor)
if [ "${PROFILE:-0}" = "1" ]; then
    ARGS+=(--profile --profile-csv "${PROFILE_CSV:-$MB_DIR/results/embedding_profile.csv}")
fi
[ -n "${EXTRA:-}" ] && read -ra EXTRA_ARR <<< "$EXTRA" && ARGS+=("${EXTRA_ARR[@]}")

mb_refresh_sudo
read -ra MODEL_ARR <<< "$MODELS"
echo "[RUN] model_bench embedding -m ${MODEL_ARR[*]} ${ARGS[*]}"
"$MB_BIN" embedding -m "${MODEL_ARR[@]}" "${ARGS[@]}"
