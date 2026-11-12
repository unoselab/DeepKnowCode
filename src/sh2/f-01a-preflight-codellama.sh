#!/usr/bin/env bash
# F-01a v1 standalone; Shell is the sole experiment parameter configuration.
set -Eeuo pipefail
export TZ=America/Chicago
export TOKENIZERS_PARALLELISM=false
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
IMPLEMENTATION="$ROOT/src/py2/f-01a-preflight-codellama.py"
PYTHON_BIN="python"
EXPERIMENT="f-01a"
VERSION="v1"
MODEL_MODE="codellama-7b"
MODEL_ID="codellama/CodeLlama-7b-hf"
# Immutable Hugging Face commit containing the original CodeLlama-7B weights.
REVISION="b9a2245269bc292b887c39da6667c631888f209e"
EXPECTED_MODEL_TYPE="llama"
EXPECTED_LAYERS=32
EXPECTED_HIDDEN=4096
EXPECTED_HEADS=32
EXPECTED_CONTEXT=16384
TARGET_MODULES=(q_proj k_proj v_proj o_proj)
ANSWER_YES=" YES"
ANSWER_NO=" NO"
PROMPT_TEXT=$'Determine whether the following two code fragments are code clones. Answer YES if they are clones and NO otherwise.\nLanguage: python\nCode A:\nprint(1)\nCode B:\nprint(1)\nAnswer:'
HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
EXPECTED_GPUS=1
MIN_VRAM_GIB=16
REQUIRE_BF16="false"
SKIP_GPU="false"
CUDA_INDEX=0
LOAD_MODE="4bit"
GROUND_TRUTH_RELATIVE="data/evaluation/ground_truth/python/verified_pairs.jsonl"
EXPECTED_GROUND_TRUTH_SHA256="2d171cdee52f6ab8229853b0cc5590b71e1a7dd6919a417dd88fb17d0ec0cb8c"
DATASET_PATHS=("$GROUND_TRUTH_RELATIVE")
LOG_DIR="$ROOT/logs/f-01a"
RESULT_DIR="$ROOT/results/f-01a"
LOG_PREFIX="f-01a-v1-$MODEL_MODE"
LOG_TIMESTAMP_FORMAT="%Y-%m%d-%H%M%S"
LOG_SAVE_ENABLED="true"
LOG_TEE_CONSOLE="true"
MODE="${1:---preflight}"
[[ $# -le 1 ]] || { echo 'Usage: --preflight|--download|--smoke' >&2; exit 2; }
case "$MODE" in
    --preflight) CMD=preflight ;;
    --download) CMD=download ;;
    --smoke) CMD=smoke ;;
    *) echo 'Usage: --preflight|--download|--smoke' >&2; exit 2 ;;
esac
[[ -f "$IMPLEMENTATION" ]] || { echo "Missing implementation $IMPLEMENTATION" >&2; exit 2; }
[[ "$LOG_SAVE_ENABLED" == true && "$LOG_TEE_CONSOLE" == true ]] || { echo 'Logging must be enabled' >&2; exit 2; }
mkdir -p -- "$LOG_DIR" "$RESULT_DIR"
STAMP="$(TZ=America/Chicago date "+$LOG_TIMESTAMP_FORMAT")"
LOG_FILE="$LOG_DIR/${LOG_PREFIX}-${CMD}-${STAMP}.log"
RESULT_FILE="$RESULT_DIR/${LOG_PREFIX}-${CMD}-${STAMP}.json"
# O_EXCL-style shell noclobber: no random suffix and no accidental overwrites.
( set -C; : > "$LOG_FILE" ) || { echo "ERROR: Log collision $LOG_FILE" >&2; exit 2; }
if [[ -e "$RESULT_FILE" ]]; then echo "ERROR: Result collision $RESULT_FILE" >&2; exit 2; fi
export HF_HOME
ARGS=( "$CMD" --experiment "$EXPERIMENT" --version "$VERSION" --model-mode "$MODEL_MODE"
 --root "$ROOT" --model-id "$MODEL_ID" --revision "$REVISION" --hf-home "$HF_HOME"
 --result-file "$RESULT_FILE" --expected-model-type "$EXPECTED_MODEL_TYPE"
 --expected-layers "$EXPECTED_LAYERS" --expected-hidden "$EXPECTED_HIDDEN"
 --expected-heads "$EXPECTED_HEADS" --expected-context "$EXPECTED_CONTEXT"
 --target-modules "${TARGET_MODULES[@]}" --answer-yes "$ANSWER_YES" --answer-no "$ANSWER_NO"
 --prompt-text "$PROMPT_TEXT" --expected-gpus "$EXPECTED_GPUS" --min-vram-gib "$MIN_VRAM_GIB"
 --require-bf16 "$REQUIRE_BF16" --skip-gpu "$SKIP_GPU" --cuda-index "$CUDA_INDEX"
 --load-mode "$LOAD_MODE" --ground-truth-relative "$GROUND_TRUTH_RELATIVE"
 --expected-ground-truth-sha256 "$EXPECTED_GROUND_TRUTH_SHA256"
 --dataset-paths "${DATASET_PATHS[@]}" )
{
  echo "[F-01a v1] MODE=$CMD MODEL_MODE=$MODEL_MODE CHICAGO=$(TZ=America/Chicago date --iso-8601=seconds)"
  echo "[F-01a v1] RESULT=$RESULT_FILE LOG=$LOG_FILE HF_HOME=$HF_HOME"
  echo "[F-01a v1] MODEL_ID=$MODEL_ID REVISION=$REVISION GPU=$CUDA_INDEX"
  "$PYTHON_BIN" -u "$IMPLEMENTATION" "${ARGS[@]}"
  echo "[F-01a v1] COMPLETE mode=$CMD"
} 2>&1 | tee -a "$LOG_FILE"
