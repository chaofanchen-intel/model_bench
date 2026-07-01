#!/bin/bash
#
# model_bench VLM (Qwen3.5) runner.
#
# Usage:
#   ./runners/run_vlm.sh CPU GPU
#   MODELS="../models/Qwen3.5-2B-INT4" IMAGE=../test_image.jpg ./runners/run_vlm.sh GPU
#
# Env overrides (defaults in []):
#   MODELS        [../models/Qwen3.5-0.8B-INT8]
#   DEVICE        [CPU]                  positional args override
#   ITERS         [5]
#   INFER_COUNT   [512]
#   IMAGE         []                     image path -> multimodal (needs MB_WITH_OPENCV build)
#   PROMPT_FILE   []                     set to a JSONL path to use a real prompt file instead of INPUT_TOKENS
#   INPUT_TOKENS  [1000]                 DEFAULT text input: synthesize a ~1000-token prompt (ignored if IMAGE/PROMPT_FILE set)
#   MONITOR       [0]
#   CSV / JSON    [results/qwen3_5.{csv,json}]
#   EXTRA         []
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MB_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
source "$SCRIPT_DIR/_common.sh"
mb_setup_env
mb_select_bin
mb_parse_devices "$@"

MODELS="${MODELS:-$MB_DIR/../models/Qwen3.5-0.8B-INT8}"
ITERS="${ITERS:-5}"
INFER_COUNT="${INFER_COUNT:-512}"
CSV="${CSV:-$MB_DIR/results/qwen3_5.csv}"
JSON="${JSON:-$MB_DIR/results/qwen3_5.json}"

# Input precedence: IMAGE (multimodal) -> PROMPT_FILE -> synthesize ~1000-token text.
# --image forces image mode in the binary, so --input-tokens is only added for the
# text-only, no-prompt-file case. INPUT_TOKENS default 1000, env-overridable.
INPUT_TOKENS="${INPUT_TOKENS:-1000}"
ARGS=(-d "${DEVICES[@]}" -n "$ITERS" -ic "$INFER_COUNT" -r "$CSV" -rj "$JSON")
if [ -n "${IMAGE:-}" ]; then
    ARGS+=(--image "$IMAGE")
elif [ -n "${PROMPT_FILE:-}" ]; then
    ARGS+=(-pf "$PROMPT_FILE")
else
    ARGS+=(--input-tokens "$INPUT_TOKENS")
fi
[ "${MONITOR:-0}" = "1" ] && ARGS+=(--monitor)
[ -n "${EXTRA:-}" ] && read -ra EXTRA_ARR <<< "$EXTRA" && ARGS+=("${EXTRA_ARR[@]}")

mb_refresh_sudo
read -ra MODEL_ARR <<< "$MODELS"
echo "[RUN] model_bench vlm -m ${MODEL_ARR[*]} ${ARGS[*]}"
"$MB_BIN" vlm -m "${MODEL_ARR[@]}" "${ARGS[@]}"
