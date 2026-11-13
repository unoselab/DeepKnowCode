#!/usr/bin/env bash
# G-01a v2: independent, shell-defined GPT-OSS-120B MXFP4 preflight and smoke.
set -Eeuo pipefail
export TZ=America/Chicago
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
PYTHON_BIN="${G01A_PYTHON_BIN:-python}"
PYTHON_SCRIPT="$ROOT/src/py2/g-01a-gptoss-preflight.py"
MODEL_ID="openai/gpt-oss-120b"
MODEL_REVISION="b5c939de8f754692c1647ca79fbf85e8c1e70f8a"
MODEL_TYPE="gpt_oss"
QUANT_METHOD="mxfp4"
SNAPSHOT="$HOME/.cache/huggingface/hub/models--openai--gpt-oss-120b/snapshots/$MODEL_REVISION"
DEVICE_IDS="0,1,2"
EXPECTED_GPUS=3
EXPECTED_SHARDS=15
MIN_SHARD_BYTES=1000000000
MIN_COMPUTE_MAJOR=8
MAX_GPU_GIB=39
MAX_CPU_GIB=4
MIN_GPU_HEADROOM_GIB=4
MAX_INPUT_TOKENS=512
MAX_NEW_TOKENS=1
ATTENTION_BACKEND="eager"
SYSTEM_TEXT="You classify code clone pairs. Reply YES or NO."
INFERENCE_TEXT="Are these code snippets clones? Code A: def f(x): return x+1; Code B: def g(y): return y+1."
ANSWER_YES=" YES"
ANSWER_NO=" NO"
EXPECTED_TORCH="2.10.0"
EXPECTED_TRANSFORMERS="4.57.6"
EXPECTED_ACCELERATE="1.15.0"
EXPECTED_KERNELS="0.12.0"
EXPECTED_TRITON="3.6.0"
EXPECTED_HUGGINGFACE_HUB="0.36.2"
ALLOW_NETWORK="true" # Only dynamically retrieved kernels; model weight loading remains local_files_only.
LOG_DIR="$ROOT/logs/g-01a/gpt-oss-120b"
RESULTS_DIR="$ROOT/results/g-01a/gpt-oss-120b"
LOG_PREFIX="g-01a-v2"
LOG_TIMESTAMP_FORMAT='+%Y-%m%d-%H%M%S'
MODE="${1:---tokenizer}"
[[ $# -le 1 ]] || { echo 'Usage: --tokenizer | --load-smoke | --inference-smoke' >&2; exit 2; }
case "$MODE" in --tokenizer|--load-smoke|--inference-smoke) ;; *) echo "Unsupported mode $MODE" >&2;exit 2;; esac
[[ -f "$PYTHON_SCRIPT" ]] || { echo "Missing $PYTHON_SCRIPT" >&2;exit 2; }
export CUDA_VISIBLE_DEVICES="$DEVICE_IDS"
export TOKENIZERS_PARALLELISM=false
mkdir -p "$LOG_DIR" "$RESULTS_DIR"
STAMP="$(TZ=America/Chicago date "$LOG_TIMESTAMP_FORMAT")"
UNIQUE="$(date +%s)-$$-${RANDOM}"
LOG_FILE="$LOG_DIR/${LOG_PREFIX}-${MODE#--}-${STAMP}-${UNIQUE}.log"
RUN_DIR="$RESULTS_DIR/${STAMP}-${UNIQUE}-${MODE#--}"
( set -C; : > "$LOG_FILE" ) || { echo "Refusing to overwrite $LOG_FILE" >&2;exit 2; }
[[ ! -e "$RUN_DIR" ]] || { echo "Refusing to overwrite $RUN_DIR" >&2;exit 2; }
exec > >(tee -a "$LOG_FILE") 2>&1
START_SEC="$(date +%s)"
finish() {
  rc=$?
  trap - EXIT
  printf '[G-01a v2] FINAL_STATUS=%s EXIT=%s CHICAGO=%s ELAPSED_SECONDS=%s RUN=%s LOG=%s\n' \
    "$([[ "$rc" == 0 ]] && echo PASS || echo FAIL)" "$rc" "$(TZ=America/Chicago date -Is)" "$(( $(date +%s) - START_SEC ))" "$RUN_DIR" "$LOG_FILE"
  exit "$rc"
}
trap finish EXIT
printf '[G-01a v2] START MODE=%s MODEL=%s REVISION=%s CHICAGO=%s\n' "$MODE" "$MODEL_ID" "$MODEL_REVISION" "$(date -Is)"
printf '[G-01a v2] ENV=%s GPU_IDS=%s MEM_GPU_GIB=%s MEM_CPU_GIB=%s LOG=%s\n' "${CONDA_DEFAULT_ENV:-unknown}" "$DEVICE_IDS" "$MAX_GPU_GIB" "$MAX_CPU_GIB" "$LOG_FILE"
[[ "${CONDA_DEFAULT_ENV:-}" == 'deepknowcode-gptoss' ]] || { echo 'ERROR: activate conda env deepknowcode-gptoss' >&2; exit 2; }
ARGS=(
 --model-id "$MODEL_ID" --revision "$MODEL_REVISION" --model-type "$MODEL_TYPE"
 --quant-method "$QUANT_METHOD" --snapshot "$SNAPSHOT" --run-dir "$RUN_DIR"
 --device-ids "$DEVICE_IDS" --expected-gpus "$EXPECTED_GPUS"
 --expected-shards "$EXPECTED_SHARDS" --min-shard-bytes "$MIN_SHARD_BYTES"
 --min-compute-major "$MIN_COMPUTE_MAJOR" --max-gpu-gib "$MAX_GPU_GIB"
 --max-cpu-gib "$MAX_CPU_GIB" --min-gpu-headroom-gib "$MIN_GPU_HEADROOM_GIB"
 --max-input-tokens "$MAX_INPUT_TOKENS" --max-new-tokens "$MAX_NEW_TOKENS"
 --expected-torch "$EXPECTED_TORCH" --expected-transformers "$EXPECTED_TRANSFORMERS"
 --expected-accelerate "$EXPECTED_ACCELERATE" --expected-kernels "$EXPECTED_KERNELS"
 --expected-triton "$EXPECTED_TRITON" --expected-huggingface-hub "$EXPECTED_HUGGINGFACE_HUB"
 --attention-backend "$ATTENTION_BACKEND" --system-text "$SYSTEM_TEXT"
 --inference-text "$INFERENCE_TEXT" --answer-yes "$ANSWER_YES" --answer-no "$ANSWER_NO"
 --allow-network "$ALLOW_NETWORK"
)
printf '[G-01a v2] STEP=%s START CHICAGO=%s\n' "${MODE#--}" "$(date -Is)"
STEP_START="$(date +%s)"
"$PYTHON_BIN" -u "$PYTHON_SCRIPT" "${MODE#--}" "${ARGS[@]}"
printf '[G-01a v2] STEP=%s PASS ELAPSED_SECONDS=%s CHICAGO=%s\n' "${MODE#--}" "$(( $(date +%s) - STEP_START ))" "$(date -Is)"
