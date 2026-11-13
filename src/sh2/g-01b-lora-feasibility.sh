#!/usr/bin/env bash
# G-01b v1 - standalone MXFP4/Harmony/source-only LoRA feasibility (NO full M1).
set -Eeuo pipefail
export TZ=America/Chicago
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
PYTHON_BIN="python"
PYFILE="$ROOT/src/py2/g-01b-gptoss-feasibility.py"
# Exact pinned model and local environment
MODEL_ID="openai/gpt-oss-120b"
MODEL_REVISION="b5c939de8f754692c1647ca79fbf85e8c1e70f8a"
MODEL_TYPE="gpt_oss"
QUANT_METHOD="mxfp4"
SNAPSHOT="$HOME/.cache/huggingface/hub/models--openai--gpt-oss-120b/snapshots/$MODEL_REVISION"
CONDA_ENV="deepknowcode-gptoss"
DEVICE_IDS="0,1,2"
EXPECTED_GPUS=3
MIN_COMPUTE_MAJOR=8
MAX_GPU_GIB=39
MAX_CPU_GIB=4
MIN_HEADROOM_GIB=4
EXPECTED_SHARDS=15
MIN_SHARD_BYTES=1000000000
ATTENTION_BACKEND="eager"
DEVICE_MAP="balanced"
PRECISION="bf16"
ALLOW_NETWORK="true" # Kernels may resolve cached/downloadable JIT artifacts; weight snapshot is pinned local.
EXPECTED_TORCH="2.10.0"
EXPECTED_TRANSFORMERS="4.57.6"
EXPECTED_PEFT="0.21.2"
EXPECTED_ACCELERATE="1.15.0"
EXPECTED_KERNELS="0.12.0"
EXPECTED_TRITON="3.6.0"
EXPECTED_HUGGINGFACE_HUB="0.36.2"
# Frozen F-01c source-domain subset: no dataset generation or selection for full M1.
FROZEN_PREPARED="$ROOT/results/f-01c/codellama-7b/2026-1112-115525-prepare/prepared"
FROZEN_TRAIN_SHA256="e2ec9d2cccc0adb0b73f81f60b3041b5a7f4fa519729f501c1cdc583c1e9b564"
FROZEN_VAL_SHA256="4404f90262c1021917ce31f0c454f87a5a634a0875307b8e84bbd51abefc752a"
FROZEN_MANIFEST_SHA256="149b9439b8eeb2d5dbdf81931cd621b962e02066de0467ab330ce2a7fd933579"
SOURCE_CODES="$ROOT/data/training/clean/java_source/data.jsonl"
FROZEN_TRAIN_COUNT=19464
FROZEN_VAL_COUNT=512
DATASET_TRAIN_COUNT=12000
DATASET_VAL_COUNT=412
MAX_SEQ_LENGTH=2048
SMOKE_SEQ_CAP=384
AUDIT_PROGRESS_EVERY=1000
# Original F series optimization settings used only for micro smoke (not full training).
SEED=2026
ANSWER_YES=" YES"
ANSWER_NO=" NO"
SYSTEM_TEXT="You are a code clone classifier. Return exactly YES or NO in the final answer."
USER_PREFIX=$'Determine whether the following two code fragments are code clones. Answer YES if they are clones and NO otherwise.\nLanguage: java\nCode A:\n'
USER_SEPARATOR=$'\nCode B:\n'
USER_SUFFIX=$'\nReturn YES or NO.'
TARGET_MODULES="q_proj,k_proj,v_proj,o_proj"
MIN_TARGET_MODULES=32
LORA_RANK=16
LORA_ALPHA=32
LORA_DROPOUT=0.05
LORA_BIAS="none"
GRADIENT_CHECKPOINTING="true"
GRADIENT_CHECKPOINT_USE_REENTRANT="false"
SMOKE_STEPS=2
OPTIMIZER="adamw_torch"
LEARNING_RATE=0.0002
WEIGHT_DECAY=0.0
MAX_GRAD_NORM=1.0
# Chicago logging managed exclusively by this shell.
LOG_DIR="$ROOT/logs/g-01b/gpt-oss-120b"
RESULTS_DIR="$ROOT/results/g-01b/gpt-oss-120b"
LOG_PREFIX="g-01b-v1"
LOG_TIMESTAMP_FORMAT='+%Y-%m%d-%H%M%S'
MODE="${1:---preflight}"
[[ $# -le 1 ]] || { echo 'Usage: --preflight | --lora-smoke | --train-smoke' >&2; exit 2; }
case "$MODE" in --preflight|--lora-smoke|--train-smoke) ;; *) echo 'Blocked: only --preflight, --lora-smoke, --train-smoke are allowed; --train forbidden' >&2;exit 2;; esac
[[ -f "$PYFILE" ]] || { echo "Missing implementation: $PYFILE" >&2;exit 2; }
export CUDA_VISIBLE_DEVICES="$DEVICE_IDS"
export TOKENIZERS_PARALLELISM=false
mkdir -p "$LOG_DIR" "$RESULTS_DIR"
STAMP="$(TZ=America/Chicago date "$LOG_TIMESTAMP_FORMAT")"
UNIQUE="$(date +%s)-$$-${RANDOM}"
RUN_DIR="$RESULTS_DIR/$STAMP-$UNIQUE-${MODE#--}"
LOG_FILE="$LOG_DIR/$LOG_PREFIX-${MODE#--}-$STAMP-$UNIQUE.log"
( set -C; : > "$LOG_FILE" ) || { echo "Refusing overwrite: $LOG_FILE" >&2;exit 2; }
[[ ! -e "$RUN_DIR" ]] || { echo "Refusing overwrite: $RUN_DIR" >&2;exit 2; }
exec > >(tee -a "$LOG_FILE") 2>&1
START_SEC="$(date +%s)"
finish() {
  local rc=$?
  trap - EXIT
  printf '[G-01b v1] FINAL_STATUS=%s EXIT=%s CHICAGO=%s DURATION_SECONDS=%s RUN=%s LOG=%s\n' \
    "$([[ "$rc" == 0 ]] && echo PASS || echo FAIL)" "$rc" "$(TZ=America/Chicago date -Is)" \
    "$(( $(date +%s) - START_SEC ))" "$RUN_DIR" "$LOG_FILE"
  exit "$rc"
}
trap finish EXIT
step_failed() { local rc=$?; printf '[G-01b v1] STEP=%s FAIL EXIT=%s CHICAGO=%s\n' "${MODE#--}" "$rc" "$(date -Is)"; exit "$rc"; }
trap step_failed ERR
printf '[G-01b v1] START MODE=%s MODEL=%s REVISION=%s CHICAGO=%s LOG=%s RUN=%s\n' "$MODE" "$MODEL_ID" "$MODEL_REVISION" "$(date -Is)" "$LOG_FILE" "$RUN_DIR"
[[ "${CONDA_DEFAULT_ENV:-}" == "$CONDA_ENV" ]] || { echo "Activate conda env $CONDA_ENV" >&2;exit 2; }
ARGS=(
 --root "$ROOT" --run-dir "$RUN_DIR" --model-id "$MODEL_ID" --revision "$MODEL_REVISION"
 --model-type "$MODEL_TYPE" --quant-method "$QUANT_METHOD" --snapshot "$SNAPSHOT"
 --device-ids "$DEVICE_IDS" --expected-gpus "$EXPECTED_GPUS" --min-compute-major "$MIN_COMPUTE_MAJOR"
 --max-gpu-gib "$MAX_GPU_GIB" --max-cpu-gib "$MAX_CPU_GIB" --min-headroom-gib "$MIN_HEADROOM_GIB"
 --expected-shards "$EXPECTED_SHARDS" --min-shard-bytes "$MIN_SHARD_BYTES"
 --attention-backend "$ATTENTION_BACKEND" --device-map "$DEVICE_MAP" --precision "$PRECISION"
 --allow-network "$ALLOW_NETWORK" --expected-torch "$EXPECTED_TORCH"
 --expected-transformers "$EXPECTED_TRANSFORMERS" --expected-peft "$EXPECTED_PEFT"
 --expected-accelerate "$EXPECTED_ACCELERATE" --expected-kernels "$EXPECTED_KERNELS"
 --expected-triton "$EXPECTED_TRITON" --expected-huggingface-hub "$EXPECTED_HUGGINGFACE_HUB"
 --frozen-prepared "$FROZEN_PREPARED" --source-codes "$SOURCE_CODES"
 --frozen-train-sha256 "$FROZEN_TRAIN_SHA256" --frozen-val-sha256 "$FROZEN_VAL_SHA256"
 --frozen-manifest-sha256 "$FROZEN_MANIFEST_SHA256"
 --frozen-train-count "$FROZEN_TRAIN_COUNT" --frozen-val-count "$FROZEN_VAL_COUNT"
 --dataset-train-count "$DATASET_TRAIN_COUNT" --dataset-val-count "$DATASET_VAL_COUNT"
 --max-seq-length "$MAX_SEQ_LENGTH" --smoke-seq-cap "$SMOKE_SEQ_CAP"
 --audit-progress-every "$AUDIT_PROGRESS_EVERY"
 --seed "$SEED" --answer-yes "$ANSWER_YES" --answer-no "$ANSWER_NO"
 --system-text "$SYSTEM_TEXT" --user-prefix "$USER_PREFIX" --user-separator "$USER_SEPARATOR"
 --user-suffix "$USER_SUFFIX" --target-modules "$TARGET_MODULES"
 --min-target-modules "$MIN_TARGET_MODULES" --lora-rank "$LORA_RANK" --lora-alpha "$LORA_ALPHA"
 --lora-dropout "$LORA_DROPOUT" --lora-bias "$LORA_BIAS"
 --gradient-checkpointing "$GRADIENT_CHECKPOINTING"
 --gradient-checkpoint-use-reentrant "$GRADIENT_CHECKPOINT_USE_REENTRANT"
 --smoke-steps "$SMOKE_STEPS" --optimizer "$OPTIMIZER" --learning-rate "$LEARNING_RATE"
 --weight-decay "$WEIGHT_DECAY" --max-grad-norm "$MAX_GRAD_NORM"
)
STEP_START="$(date +%s)"
printf '[G-01b v1] STEP=%s START CHICAGO=%s\n' "${MODE#--}" "$(TZ=America/Chicago date -Is)"
"$PYTHON_BIN" -u "$PYFILE" "${MODE#--}" "${ARGS[@]}"
printf '[G-01b v1] STEP=%s PASS ELAPSED_SECONDS=%s CHICAGO=%s\n' "${MODE#--}" "$(( $(date +%s) - STEP_START ))" "$(TZ=America/Chicago date -Is)"
