#!/bin/bash
#
# model_bench Qwen3-Reranker runner.
#
# Usage:
#   ./runners/run_reranker.sh CPU GPU NPU
#   MODELS="../models/Qwen3-Reranker-0.6B-int8-ov" NUM_DOCS=16 ITERS=100 ./runners/run_reranker.sh GPU
#
# Env overrides (defaults in []):
#   MODELS        [../models/Qwen3-Reranker-0.6B-int8-ov]
#   DEVICE        [CPU]
#   ITERS         [5]
#   TOP_N         [3]
#   NUM_DOCS      []                 --num-documents (replicate/trim candidate set)
#   TEXTS_FILE    []                 JSONL documents file
#   MAX_LENGTH    []
#   PROFILE       [0]                1 -> raw-Core per-op graph profiling
#   PROFILE_CSV   [results/reranker_profile.csv when PROFILE=1]
#   MONITOR       [0]
#   CSV / JSON    [results/reranker.{csv,json}]
#   EXTRA         []
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MB_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
source "$SCRIPT_DIR/_common.sh"
mb_setup_env
mb_select_bin
mb_parse_devices "$@"

MODELS="${MODELS:-$MB_DIR/../models/Qwen3-Reranker-0.6B-int8-ov}"
ITERS="${ITERS:-5}"
TOP_N="${TOP_N:-3}"
CSV="${CSV:-$MB_DIR/results/reranker.csv}"
JSON="${JSON:-$MB_DIR/results/reranker.json}"

ARGS=(-d "${DEVICES[@]}" -n "$ITERS" --top-n "$TOP_N" -r "$CSV" -rj "$JSON")
[ -n "${NUM_DOCS:-}" ] && ARGS+=(--num-documents "$NUM_DOCS")
[ -n "${TEXTS_FILE:-}" ] && ARGS+=(--texts-file "$TEXTS_FILE")
[ -n "${MAX_LENGTH:-}" ] && ARGS+=(--max-length "$MAX_LENGTH")
[ "${MONITOR:-0}" = "1" ] && ARGS+=(--monitor)
if [ "${PROFILE:-0}" = "1" ]; then
    ARGS+=(--profile --profile-csv "${PROFILE_CSV:-$MB_DIR/results/reranker_profile.csv}")
fi
[ -n "${EXTRA:-}" ] && read -ra EXTRA_ARR <<< "$EXTRA" && ARGS+=("${EXTRA_ARR[@]}")

mb_refresh_sudo
read -ra MODEL_ARR <<< "$MODELS"
echo "[RUN] model_bench reranker -m ${MODEL_ARR[*]} ${ARGS[*]}"
"$MB_BIN" reranker -m "${MODEL_ARR[@]}" "${ARGS[@]}"
