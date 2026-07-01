#!/bin/bash
#
# model_bench LLM (Qwen3) runner.
#
# Usage:
#   ./runners/run_llm.sh                 # CPU, default model
#   ./runners/run_llm.sh GPU NPU         # sweep devices
#   MODELS="../models/Qwen3-1.7B-INT4 ../models/Qwen3-4B-INT4" ITERS=10 ./runners/run_llm.sh GPU
#
# Env overrides (defaults in []):
#   MODELS        [../models/Qwen3-0.6B-INT4]  space-separated model dir(s)
#   DEVICE        [CPU]                          positional args override
#   ITERS         [5]                            measured iterations (+1 warmup)
#   INFER_COUNT   [512]                          max_new_tokens
#   PROMPT_FILE   []                             set to a JSONL path to use a real prompt file (-pf) instead of INPUT_TOKENS
#   INPUT_TOKENS  [1000]                         DEFAULT: synthesize a ~1000-token prompt (set N to change; PROMPT_FILE overrides)
#   MONITOR       [0]                            1 -> enable HW utilization monitoring
#   PROFILE       [0]                            1 -> device-property introspection
#   CSV / JSON    [results/qwen3.{csv,json}]
#   EXTRA         []                             extra args appended verbatim
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MB_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
source "$SCRIPT_DIR/_common.sh"
mb_setup_env
mb_select_bin
mb_parse_devices "$@"

MODELS="${MODELS:-$MB_DIR/../models/Qwen3-0.6B-INT4}"
ITERS="${ITERS:-5}"
INFER_COUNT="${INFER_COUNT:-512}"
CSV="${CSV:-$MB_DIR/results/qwen3.csv}"
JSON="${JSON:-$MB_DIR/results/qwen3.json}"

# Default input: synthesize ~1000-token prompt. PROMPT_FILE (if set) overrides; otherwise
# pass --input-tokens (default 1000, env-overridable via INPUT_TOKENS).
INPUT_TOKENS="${INPUT_TOKENS:-1000}"
ARGS=(-d "${DEVICES[@]}" -n "$ITERS" -ic "$INFER_COUNT" -r "$CSV" -rj "$JSON")
if [ -n "${PROMPT_FILE:-}" ]; then
    ARGS+=(-pf "$PROMPT_FILE")
else
    ARGS+=(--input-tokens "$INPUT_TOKENS")
fi
[ "${MONITOR:-0}" = "1" ] && ARGS+=(--monitor)
[ "${PROFILE:-0}" = "1" ] && ARGS+=(--profile)
[ -n "${EXTRA:-}" ] && read -ra EXTRA_ARR <<< "$EXTRA" && ARGS+=("${EXTRA_ARR[@]}")

mb_refresh_sudo
read -ra MODEL_ARR <<< "$MODELS"
echo "[RUN] model_bench llm -m ${MODEL_ARR[*]} ${ARGS[*]}"
"$MB_BIN" llm -m "${MODEL_ARR[@]}" "${ARGS[@]}"
