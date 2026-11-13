#!/usr/bin/env bash
# G-01d v4: one-command GPT-OSS-120B M1 training orchestrator.
# Invokes pinned, verified G-01d v4 Python Engine directly; NEVER calls another shell wrapper.
# All experiment configuration is defined in THIS shell and explicitly forwarded to Python.
set -Eeuo pipefail
export TZ=America/Chicago
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
PYTHON_BIN="python"
PYFILE="$ROOT/src/py2/g-01d-train-source-lora.py"
AUDITOR="$ROOT/src/py2/g-01d-pipeline-audit.py"
VERSION="v4"
ENGINE_VERSION="v4"
CONDA_ENV="deepknowcode-gptoss"
EXPECTED_ENGINE_SHA256="5e63eb94510b4aab42b93a889aef4164bddd1019655bdb6bcc10f35e31610228"
# Reuse the exact uploaded G-01c v1 prepare proof; never regenerate/select source pairs.
PREPARED_RUN="$ROOT/results/g-01c/gpt-oss-120b/2026-1112-235431-1794549271-271004-23954-prepare"
EXPECTED_PREPARED_MANIFEST_SHA256="d61894e8c55a3aa31f9bdeff7d2003fee50ff993e85a2cac9d005cb3f5049449"
EXPECTED_PREPARED_FINGERPRINT="928a08222177031206bb68e4421d9801da35d98b9973f1599abc8bbd5ceefe74"
EXPECTED_PREPARED_TRAIN_SHA256="3cf5b22e5f74191662e3607411fe19c1077c81cb180cc0bd0a11b389dd385759"
EXPECTED_PREPARED_VAL_SHA256="98856974b9f8c60df9f2dd22dc355c1bb6f4468e941bbda97e87a4c60b26827e"
EXPECTED_PREPARED_ORDER_SHA256="d2dcb384da3b22622abdb1361c15e172a7b66648582f64a75e2d180da3f9de55"
MODEL_ID="openai/gpt-oss-120b"
MODEL_REVISION="b5c939de8f754692c1647ca79fbf85e8c1e70f8a"
MODEL_TYPE="gpt_oss"
QUANT_METHOD="mxfp4"
SNAPSHOT="$HOME/.cache/huggingface/hub/models--openai--gpt-oss-120b/snapshots/$MODEL_REVISION"
# Hardware and software constraints, identical to the proven G-01b environment
DEVICE_IDS="0,1,2"
EXPECTED_GPUS=3
MIN_COMPUTE_MAJOR=8
MAX_GPU_GIB=39
MAX_CPU_GIB=4
MIN_HEADROOM_GIB=4
# Start reclaiming inactive CUDA cache before the v3 failure at ~31.88 GiB reserved.
CACHE_RECLAIM_TRIGGER_GIB=30
EXPECTED_SHARDS=15
MIN_SHARD_BYTES=1000000000
ATTENTION_BACKEND="eager"
DEVICE_MAP="balanced"
PRECISION="bf16"
ALLOW_NETWORK="true"  # MXFP4 kernels may be fetched. Base weight paths are pinned and local.
EXPECTED_TORCH="2.10.0"
EXPECTED_TRANSFORMERS="4.57.6"
EXPECTED_PEFT="0.21.2"
EXPECTED_ACCELERATE="1.15.0"
EXPECTED_KERNELS="0.12.0"
EXPECTED_TRITON="3.6.0"
EXPECTED_HUGGINGFACE_HUB="0.36.2"
# Frozen F-01c Java source subset. No new samples or held-out labels used here.
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
AUDIT_PROGRESS_EVERY=1000
# Exact G-01b verified Harmony prompt and single final-token SFT label policy.
SEED=2026
ANSWER_YES=" YES"
ANSWER_NO=" NO"
# G-01c prepared at 2026-11-12 23:56 Chicago; GPT-OSS Harmony has a
# dynamic `Current date:` via strftime_now(). Freeze the original date.
TEMPLATE_DATE="2026-11-12"
SYSTEM_TEXT="You are a code clone classifier. Return exactly YES or NO in the final answer."
USER_PREFIX=$'Determine whether the following two code fragments are code clones. Answer YES if they are clones and NO otherwise.\nLanguage: java\nCode A:\n'
USER_SEPARATOR=$'\nCode B:\n'
USER_SUFFIX=$'\nReturn YES or NO.'
TARGET_MODULES="q_proj,k_proj,v_proj,o_proj"
MIN_TARGET_MODULES=32
EXPECTED_TRAINABLE_PARAMETERS=11943936
LORA_RANK=16
LORA_ALPHA=32
LORA_DROPOUT=0.05
LORA_BIAS="none"
GRADIENT_CHECKPOINTING="true"
GRADIENT_CHECKPOINT_USE_REENTRANT="false"
# Stress/stability preconditions. Each stage loads FRESH frozen base and new adapter.
STRESS_STEPS=2
STRESS_MIN_TOKENS=1200
STABILITY_STEPS=8
# M1 exact budget: single process * batch 1 * grad accumulation 24 * steps 500 = 12,000.
PER_DEVICE_BATCH_SIZE=1
GRADIENT_ACCUMULATION_STEPS=24
MAX_STEPS=500
EXPECTED_EPOCHS=1
OPTIMIZER="adamw_torch"
LEARNING_RATE="0.0002"
WEIGHT_DECAY="0.0"
MAX_GRAD_NORM="1.0"
WARMUP_STEPS=20
LR_SCHEDULER="cosine"
LOG_STEPS=10
EVAL_STEPS=100
SAVE_STEPS=100
EVAL_BATCH_SIZE=1
SAVE_OPTIMIZER_STATE="true"

# Runtime and automatic Chicago timestamp logs.
LOG_DIR="$ROOT/logs/g-01d/gpt-oss-120b"
RESULTS_DIR="$ROOT/results/g-01d/gpt-oss-120b"
LOG_PREFIX="g-01d-v4"
LOG_TIMESTAMP_FORMAT='+%Y-%m%d-%H%M%S'
LOG_TEE_OPTIONS='-a'
[[ "$#" -eq 1 && "$1" == '--train' ]] || { echo 'Usage: bash src/sh2/g-01d-train-source-lora.sh --train' >&2; exit 2; }
[[ -f "$PYFILE" && -f "$AUDITOR" ]] || { echo 'G-01d v4 auditor or Python engine missing' >&2; exit 2; }
[[ "${CONDA_DEFAULT_ENV:-}" == "$CONDA_ENV" ]] || { echo "ERROR: conda activate $CONDA_ENV" >&2; exit 2; }
export CUDA_VISIBLE_DEVICES="$DEVICE_IDS"
export TOKENIZERS_PARALLELISM='false'
mkdir -p "$LOG_DIR" "$RESULTS_DIR"
STAMP="$(TZ=America/Chicago date "$LOG_TIMESTAMP_FORMAT")"
UNIQUE="$(date +%s)-$$-${RANDOM}"
PIPELINE="$RESULTS_DIR/$STAMP-$UNIQUE-pipeline"
LOG_FILE="$LOG_DIR/$LOG_PREFIX-train-$STAMP-$UNIQUE.log"
( set -C; : > "$LOG_FILE" ) || { echo "ERROR: refusing to overwrite $LOG_FILE" >&2; exit 2; }
[[ ! -e "$PIPELINE" ]] || { echo "ERROR: refusing to overwrite $PIPELINE" >&2; exit 2; }
mkdir "$PIPELINE"
# stdout+stderr -> terminal and unique LOG; one log covers every major step.
exec > >(tee "$LOG_TEE_OPTIONS" "$LOG_FILE") 2>&1
START_SEC="$(date +%s)"
ACTIVE_STEP='initialization'
step_start_time=0
at_chicago() { TZ=America/Chicago date -Is; }
finish() {
  local rc=$?
  trap - EXIT ERR
  if [[ $rc -ne 0 ]]; then
    "$PYTHON_BIN" "$AUDITOR" final-report --pipeline "$PIPELINE" --status FAIL \
      --step "$ACTIVE_STEP" --exit-code "$rc" --log-file "$LOG_FILE" \
      --prepared-run "$PREPARED_RUN" --fingerprint "$EXPECTED_PREPARED_FINGERPRINT" \
      --elapsed-seconds "$(( $(date +%s) - START_SEC ))" || true
  fi
  printf '[G-01d %s] FINAL_STATUS=%s EXIT=%s STEP=%s CHICAGO=%s DURATION_SECONDS=%s PREPARED=%s RUN=%s LOG=%s\n' \
    "$VERSION" "$( [[ $rc -eq 0 ]] && echo PASS || echo FAIL )" "$rc" \
    "$ACTIVE_STEP" "$(at_chicago)" "$(( $(date +%s) - START_SEC ))" \
    "$PREPARED_RUN" "$PIPELINE" "$LOG_FILE"
  exit "$rc"
}
trap finish EXIT
trap 'rc=$?; printf "[G-01d %s] STEP=%s FAIL EXIT=%s CHICAGO=%s\n" "$VERSION" "$ACTIVE_STEP" "$rc" "$(at_chicago)"; exit "$rc"' ERR
step_begin() {
  ACTIVE_STEP="$1"
  step_start_time="$(date +%s)"
  printf '[G-01d %s] STEP=%s START CHICAGO=%s\n' "$VERSION" "$ACTIVE_STEP" "$(at_chicago)"
}
step_pass() {
  printf '[G-01d %s] STEP=%s PASS DURATION_SECONDS=%s CHICAGO=%s\n' "$VERSION" "$ACTIVE_STEP" "$(( $(date +%s) - step_start_time ))" "$(at_chicago)"
}
printf '[G-01d %s] START MODE=--train MODEL=%s REVISION=%s CHICAGO=%s PIPELINE=%s LOG=%s\n' \
  "$VERSION" "$MODEL_ID" "$MODEL_REVISION" "$(at_chicago)" "$PIPELINE" "$LOG_FILE"
printf '[G-01d %s] ENGINE=G-01d v4 Python (direct execution, no shell-wrapper chaining) TEMPLATE_DATE=%s\n' "$VERSION" "$TEMPLATE_DATE"

# Shared training-engine parameters; set mode/run/proof paths for each step below.
RUN_DIR=''
STRESS_RUN=''
STABILITY_RUN=''
RESUME_CHECKPOINT=''
build_args() {
  ARGS=(
  --version "$VERSION" --root "$ROOT" --run-dir "$RUN_DIR" --log-file "$LOG_FILE"
  --prepared-run "$PREPARED_RUN" --stress-run "$STRESS_RUN" --stability-run "$STABILITY_RUN"
  --resume-checkpoint "$RESUME_CHECKPOINT"
  --model-id "$MODEL_ID" --revision "$MODEL_REVISION" --model-type "$MODEL_TYPE"
  --quant-method "$QUANT_METHOD" --snapshot "$SNAPSHOT"
  --device-ids "$DEVICE_IDS" --expected-gpus "$EXPECTED_GPUS" --min-compute-major "$MIN_COMPUTE_MAJOR"
  --max-gpu-gib "$MAX_GPU_GIB" --max-cpu-gib "$MAX_CPU_GIB" --min-headroom-gib "$MIN_HEADROOM_GIB"
  --cache-reclaim-trigger-gib "$CACHE_RECLAIM_TRIGGER_GIB"
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
  --max-seq-length "$MAX_SEQ_LENGTH" --audit-progress-every "$AUDIT_PROGRESS_EVERY"
  --seed "$SEED" --answer-yes "$ANSWER_YES" --answer-no "$ANSWER_NO"
  --system-text "$SYSTEM_TEXT" --user-prefix "$USER_PREFIX" --user-separator "$USER_SEPARATOR"
  --user-suffix "$USER_SUFFIX" --template-date "$TEMPLATE_DATE" --target-modules "$TARGET_MODULES"
  --min-target-modules "$MIN_TARGET_MODULES" --expected-trainable-parameters "$EXPECTED_TRAINABLE_PARAMETERS" --lora-rank "$LORA_RANK" --lora-alpha "$LORA_ALPHA"
  --lora-dropout "$LORA_DROPOUT" --lora-bias "$LORA_BIAS"
  --gradient-checkpointing "$GRADIENT_CHECKPOINTING"
  --gradient-checkpoint-use-reentrant "$GRADIENT_CHECKPOINT_USE_REENTRANT"
  --stress-steps "$STRESS_STEPS" --stress-min-tokens "$STRESS_MIN_TOKENS"
  --stability-steps "$STABILITY_STEPS" --per-device-batch-size "$PER_DEVICE_BATCH_SIZE"
  --gradient-accumulation-steps "$GRADIENT_ACCUMULATION_STEPS" --max-steps "$MAX_STEPS"
  --expected-epochs "$EXPECTED_EPOCHS" --optimizer "$OPTIMIZER" --learning-rate "$LEARNING_RATE"
  --weight-decay "$WEIGHT_DECAY" --max-grad-norm "$MAX_GRAD_NORM"
  --warmup-steps "$WARMUP_STEPS" --lr-scheduler "$LR_SCHEDULER" --log-steps "$LOG_STEPS"
  --eval-steps "$EVAL_STEPS" --save-steps "$SAVE_STEPS" --eval-batch-size "$EVAL_BATCH_SIZE"
  --save-optimizer-state "$SAVE_OPTIMIZER_STATE"
  )
}

# 1. Early immutable identity audit; fail before allocating 120B GPU weights.
step_begin 'PREPARED_PROOF_AND_ENGINE'
"$PYTHON_BIN" -u "$AUDITOR" prepare \
  --prepared-run "$PREPARED_RUN" \
  --engine "$PYFILE" --expected-engine-sha256 "$EXPECTED_ENGINE_SHA256" \
  --expected-manifest-sha256 "$EXPECTED_PREPARED_MANIFEST_SHA256" \
  --expected-fingerprint "$EXPECTED_PREPARED_FINGERPRINT" \
  --expected-train-sha256 "$EXPECTED_PREPARED_TRAIN_SHA256" \
  --expected-val-sha256 "$EXPECTED_PREPARED_VAL_SHA256" \
  --expected-order-sha256 "$EXPECTED_PREPARED_ORDER_SHA256" \
  --expected-train-count "$DATASET_TRAIN_COUNT" \
  --expected-val-count "$DATASET_VAL_COUNT" \
  --expected-steps "$MAX_STEPS" --expected-global-batch "$GRADIENT_ACCUMULATION_STEPS" \
  --expected-epochs "$EXPECTED_EPOCHS" \
  --frozen-train-sha256 "$FROZEN_TRAIN_SHA256" \
  --frozen-val-sha256 "$FROZEN_VAL_SHA256" \
  --frozen-manifest-sha256 "$FROZEN_MANIFEST_SHA256"
step_pass

# Helper to update mode-specific dynamic CLI args, run G-01d Python directly,
# then verify evidence. Each run directory is exclusive and never overwritten.
run_stage() {
  local mode="$1" ordinal="$2" expected_status="$3" steps_expected="$4"
  RUN_DIR="$PIPELINE/$ordinal-$mode"
  [[ ! -e "$RUN_DIR" ]] || { echo "ERROR: existing run $RUN_DIR" >&2; return 2; }
  mkdir "$RUN_DIR"
  step_begin "$mode"
  build_args
  "$PYTHON_BIN" -u "$PYFILE" "$mode" "${ARGS[@]}"
  "$PYTHON_BIN" -u "$AUDITOR" stage \
    --stage-run "$RUN_DIR" --prepared-run "$PREPARED_RUN" \
    --expected-fingerprint "$EXPECTED_PREPARED_FINGERPRINT" \
    --expected-manifest-sha256 "$EXPECTED_PREPARED_MANIFEST_SHA256" \
    --expected-status "$expected_status" --expected-steps "$steps_expected" \
    --max-gpu-gib "$MAX_GPU_GIB" --min-headroom-gib "$MIN_HEADROOM_GIB" \
    --cache-reclaim-trigger-gib "$CACHE_RECLAIM_TRIGGER_GIB" \
    --expected-model-revision "$MODEL_REVISION" \
    --expected-train-count "$DATASET_TRAIN_COUNT" --expected-val-count "$DATASET_VAL_COUNT"
  step_pass
}

STRESS_RUN=''
STABILITY_RUN=''
run_stage stress '01' 'G01D_V4_STRESS_PASS_NOT_M1_TRAINING' "$STRESS_STEPS"
STRESS_RUN="$RUN_DIR"
run_stage stability '02' 'G01D_V4_STABILITY_PASS_NOT_M1_TRAINING' "$STABILITY_STEPS"
STABILITY_RUN="$RUN_DIR"
run_stage train '03' 'G01D_V4_M1_SOURCE_ONLY_TRAINING_COMPLETE' "$MAX_STEPS"
TRAIN_RUN="$RUN_DIR"

step_begin 'FINAL_ADAPTER_VERIFY'
"$PYTHON_BIN" -u "$AUDITOR" final-report \
  --pipeline "$PIPELINE" --status PASS --step 'FINAL_ADAPTER_VERIFY' \
  --exit-code 0 --log-file "$LOG_FILE" --prepared-run "$PREPARED_RUN" \
  --fingerprint "$EXPECTED_PREPARED_FINGERPRINT" \
  --elapsed-seconds "$(( $(date +%s) - START_SEC ))" \
  --train-run "$TRAIN_RUN" --stress-run "$STRESS_RUN" --stability-run "$STABILITY_RUN" \
  --expected-manifest-sha256 "$EXPECTED_PREPARED_MANIFEST_SHA256" \
  --expected-steps "$MAX_STEPS" --expected-val-events "$(( MAX_STEPS / EVAL_STEPS ))"
step_pass
ACTIVE_STEP='completed'
