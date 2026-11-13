#!/usr/bin/env bash
# G-02 v2: standalone GPT-OSS-120B M2 Standard Mixed LoRA, M1 adapter continuation.
# No calls to G-01d/F-Series shell wrappers. All configuration is defined here.
# --prepare performs read-only frozen-dataset/model-tokenizer audit, no 120B weight loading.
# --train performs prepare -> stress -> stability -> full M2 -> final audit.
set -Eeuo pipefail
export TZ=America/Chicago
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
PYTHON_BIN="python"
PYFILE="$ROOT/src/py2/g-02-train-standard-mixed-lora.py"
AUDITOR="$ROOT/src/py2/g-02-pipeline-audit.py"
VERSION="v2"
CONDA_ENV="deepknowcode-gptoss"

MODEL_ID="openai/gpt-oss-120b"
REVISION="b5c939de8f754692c1647ca79fbf85e8c1e70f8a"
MODEL_TYPE="gpt_oss"
QUANT_METHOD="mxfp4"
SNAPSHOT="$HOME/.cache/huggingface/hub/models--openai--gpt-oss-120b/snapshots/$REVISION"
DEVICE_IDS="0,1,2"
EXPECTED_GPUS=3
MIN_COMPUTE_MAJOR=8
MAX_GPU_GIB=39
MAX_CPU_GIB=4
MIN_HEADROOM_GIB=4
CACHE_RECLAIM_TRIGGER_GIB=30
EXPECTED_SHARDS=15
MIN_SHARD_BYTES=1000000000
ATTENTION_BACKEND="eager"
DEVICE_MAP="balanced"
PRECISION="bf16"
ALLOW_NETWORK="true"
EXPECTED_TORCH="2.10.0"
EXPECTED_TRANSFORMERS="4.57.6"
EXPECTED_PEFT="0.21.2"
EXPECTED_ACCELERATE="1.15.0"
EXPECTED_KERNELS="0.12.0"
EXPECTED_TRITON="3.6.0"
EXPECTED_HUGGINGFACE_HUB="0.36.2"

# Actual completed G-01d v4 M1 adapter; no fresh randomly initialized LoRA in M2.
M1_RUN="$ROOT/results/g-01d/gpt-oss-120b/2026-1113-004452-1794552292-275033-14249-pipeline/03-train"
M1_ADAPTER_SHA256="0c9d913781a8a06342fef64113bfbb2ccd2d9d7dddd2b41d760915dbf8018cbf"
M1_ADAPTER_MANIFEST_SHA256="454293ad3495e657c4f926650814d7bf52ab36e357d8839b784724f5d25b8baf"
M1_SUMMARY_SHA256="5e0a60c743f41d7e9e7588066a743d67c3ab8e01e84f68aab2f41011d0d36c52"
M1_SOURCE_MANIFEST_SHA256="d61894e8c55a3aa31f9bdeff7d2003fee50ff993e85a2cac9d005cb3f5049449"
M1_OPTIMIZER_STEPS=500
SOURCE_PREPARED="$ROOT/results/g-01c/gpt-oss-120b/2026-1112-235431-1794549271-271004-23954-prepare/prepared"
SOURCE_PREPARED_TRAIN_SHA256="3cf5b22e5f74191662e3607411fe19c1077c81cb180cc0bd0a11b389dd385759"
SOURCE_PREPARED_VAL_SHA256="98856974b9f8c60df9f2dd22dc355c1bb6f4468e941bbda97e87a4c60b26827e"

# F-01c v24 frozen Standard Mixed rows, not dynamically sampled/modified.
FROZEN_PREPARED="$ROOT/results/f-01c/codellama-7b/2026-1112-115525-prepare/prepared"
FROZEN_TRAIN_SHA256="e2ec9d2cccc0adb0b73f81f60b3041b5a7f4fa519729f501c1cdc583c1e9b564"
FROZEN_VAL_SHA256="4404f90262c1021917ce31f0c454f87a5a634a0875307b8e84bbd51abefc752a"
FROZEN_MANIFEST_SHA256="149b9439b8eeb2d5dbdf81931cd621b962e02066de0467ab330ce2a7fd933579"
SOURCE_CODES="$ROOT/data/training/clean/java_source/data.jsonl"
FROZEN_TRAIN_COUNT=19464
FROZEN_VAL_COUNT=512
DATASET_TRAIN_COUNT=19464
DATASET_VAL_COUNT=512
SOURCE_TRAIN_COUNT=12000
JAVA_TRAIN_COUNT=3732
PYTHON_TRAIN_COUNT=3732
# These validation quotas are immutable inputs from the SHA256-pinned F-01c manifest.
# The shell resolves and passes all three numerically; the engine never guesses them.
FROZEN_VAL_QUOTA_POLICY="SHA256_PINNED_F01C_MANIFEST"
SOURCE_VAL_COUNT=""
JAVA_VAL_COUNT=""
PYTHON_VAL_COUNT=""
# F01c frozen manifest was PREPARED_PENDING_LABEL_REVIEW. Explicitly disclose it.
# This policy binds to the ALREADY USED F01c Frozen Study Dataset, not a new label endorsement.
PENDING_LABEL_REVIEW_POLICY="EXISTING_F01C_FROZEN_PROTOCOL_WITH_DISCLOSED_RISK"
MAX_SEQ_LENGTH=2048
AUDIT_PROGRESS_EVERY=1000
AUDIT_FAILURE_EXAMPLES=40
SEED=2026
ANSWER_YES=" YES"
ANSWER_NO=" NO"
# Retain M1 G01c Harmony date to avoid GPTOSS current-date prompt identity drift.
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
LORA_DROPOUT="0.05"
LORA_BIAS="none"
GRADIENT_CHECKPOINTING="true"
GRADIENT_CHECKPOINT_USE_REENTRANT="false"
STRESS_STEPS=2
STRESS_MIN_TOKENS=1200
STABILITY_STEPS=8
PER_DEVICE_BATCH_SIZE=1
GRADIENT_ACCUMULATION_STEPS=24
MAX_STEPS=811
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

LOG_DIR="$ROOT/logs/g-02/gpt-oss-120b"
RESULTS_DIR="$ROOT/results/g-02/gpt-oss-120b"
LOG_PREFIX="g-02-v1"
LOG_TIMESTAMP_FORMAT='+%Y-%m%d-%H%M%S'
LOG_TEE_OPTIONS='-a'
[[ $# -eq 1 && ( "$1" == '--train' || "$1" == '--prepare' ) ]] || {
  echo "Usage: bash src/sh2/g-02-train-standard-mixed-lora.sh --prepare|--train" >&2;exit 2;
}
MODE="$1"
[[ -f "$PYFILE" && -f "$AUDITOR" ]] || { echo 'Missing standalone G-02 Python engine/auditor' >&2;exit 2; }
[[ "${CONDA_DEFAULT_ENV:-}" == "$CONDA_ENV" ]] || { echo "ERROR: conda activate $CONDA_ENV" >&2;exit 2; }
export CUDA_VISIBLE_DEVICES="$DEVICE_IDS" TOKENIZERS_PARALLELISM=false
mkdir -p "$LOG_DIR" "$RESULTS_DIR"
STAMP="$(TZ=America/Chicago date "$LOG_TIMESTAMP_FORMAT")"
UNIQUE="$(date +%s)-$$-$RANDOM"
PIPELINE="$RESULTS_DIR/$STAMP-$UNIQUE-pipeline"
LOG_FILE="$LOG_DIR/$LOG_PREFIX-${MODE#--}-$STAMP-$UNIQUE.log"
( set -C; : > "$LOG_FILE" ) || { echo 'Refusing to overwrite log file' >&2;exit 2; }
[[ ! -e "$PIPELINE" ]] || { echo 'Refusing to overwrite run path' >&2;exit 2; }
mkdir "$PIPELINE"
exec > >(tee "$LOG_TEE_OPTIONS" "$LOG_FILE") 2>&1
START_SEC="$(date +%s)"
ACTIVE_STEP='init'
RUN_DIR=''
PREPARED_RUN=''
STRESS_RUN=''
STABILITY_RUN=''
RESUME_CHECKPOINT=''
EXPECTED_FINGERPRINT=''
at_chicago() { TZ=America/Chicago date -Is; }
finish() {
  local rc=$?
  trap - EXIT ERR
  if [[ ! -f "$PIPELINE/pipeline_result.json" ]]; then
    "$PYTHON_BIN" -u "$AUDITOR" finish --pipeline "$PIPELINE" \
      --stage-run "$RUN_DIR" --status '' --prepared-run "$PREPARED_RUN" \
      --expected-fingerprint "$EXPECTED_FINGERPRINT" \
      --m1-adapter-sha256 "$M1_ADAPTER_SHA256" --log-file "$LOG_FILE" \
      --step "$ACTIVE_STEP" --steps "$MAX_STEPS" --exit-code "$rc" \
      --max-gpu-gib "$MAX_GPU_GIB" --min-headroom-gib "$MIN_HEADROOM_GIB" \
      --operation="$MODE" || true
  fi
  printf '[G-02 %s] FINAL_STATUS=%s EXIT=%s STEP=%s CHICAGO=%s DURATION_SECONDS=%s PIPELINE=%s LOG=%s\n' \
    "$VERSION" "$( [[ $rc -eq 0 ]] && echo PASS || echo FAIL )" "$rc" \
    "$ACTIVE_STEP" "$(at_chicago)" "$(( $(date +%s) - START_SEC ))" "$PIPELINE" "$LOG_FILE"
  exit "$rc"
}
trap finish EXIT
step() {
  local stage="$1" start rc
  shift; ACTIVE_STEP="$stage"; start="$(date +%s)"
  printf '[G-02 %s] STEP=%s START CHICAGO=%s\n' "$VERSION" "$stage" "$(at_chicago)"
  if "$@"; then
    printf '[G-02 %s] STEP=%s PASS ELAPSED_SECONDS=%s CHICAGO=%s\n' "$VERSION" "$stage" \
      "$(( $(date +%s) - start ))" "$(at_chicago)"
  else
    rc=$?
    printf '[G-02 %s] STEP=%s FAIL EXIT=%s ELAPSED_SECONDS=%s CHICAGO=%s\n' \
      "$VERSION" "$stage" "$rc" "$(( $(date +%s) - start ))" "$(at_chicago)"
    return "$rc"
  fi
}


# Only the SHA256-pinned F01c manifest may define validation-domain quotas.
# The experiment policy is explicit in this shell and each resolved numeric value
# is passed to the Python engine as a required CLI argument; no Python defaults.
resolve_frozen_val_quotas() {
  local quota_line
  quota_line="$("$PYTHON_BIN" - "$FROZEN_PREPARED/manifest.json" \
    "$FROZEN_MANIFEST_SHA256" "$FROZEN_VAL_COUNT" \
    "$SOURCE_TRAIN_COUNT" "$JAVA_TRAIN_COUNT" "$PYTHON_TRAIN_COUNT" <<'PYQUOTA'
import hashlib, json, pathlib, sys
path=pathlib.Path(sys.argv[1])
expected=sys.argv[2]
expected_validation=int(sys.argv[3])
expected_train=list(map(int,sys.argv[4:7]))
if not path.is_file():
    raise SystemExit('G02 frozen M2 manifest is missing: '+str(path))
raw=path.read_bytes()
actual=hashlib.sha256(raw).hexdigest()
if actual!=expected:
    raise SystemExit('G02 frozen M2 manifest SHA256 mismatch: '+actual)
m=json.loads(raw)
if m.get('status')!='PREPARED_PENDING_LABEL_REVIEW':
    raise SystemExit('G02 unexpected frozen manifest status')
plan=m.get('training_plan',{})
keys=('java_source','java_mixed','python_mixed')
for split in ('train','validation'):
    obj=plan.get(split)
    if not isinstance(obj,dict) or set(obj)!=set(keys) or any(type(obj[x]) is not int or obj[x]<=0 or obj[x]%2 for x in keys):
        raise SystemExit('G02 invalid frozen per-domain '+split+' quotas: '+repr(obj))
train=[plan['train'][x] for x in keys]
val=[plan['validation'][x] for x in keys]
if train!=expected_train:
    raise SystemExit('G02 frozen training quotas differ from Shell: actual='+repr(train)+' expected='+repr(expected_train))
if sum(val)!=expected_validation:
    raise SystemExit('G02 frozen validation total differs from Shell: actual='+str(sum(val)))
print(*val)
PYQUOTA
)" || return 2
  read -r SOURCE_VAL_COUNT JAVA_VAL_COUNT PYTHON_VAL_COUNT <<< "$quota_line"
  [[ "$SOURCE_VAL_COUNT" =~ ^[0-9]+$ && "$JAVA_VAL_COUNT" =~ ^[0-9]+$ && "$PYTHON_VAL_COUNT" =~ ^[0-9]+$ ]] || {
    echo 'ERROR: malformed pinned frozen validation quota result' >&2; return 2;
  }
  printf '[G-02 %s] FROZEN_QUOTA_PIN_PASS POLICY=%s TRAIN=%s,%s,%s VALIDATION=%s,%s,%s SHA256=%s\n' \
    "$VERSION" "$FROZEN_VAL_QUOTA_POLICY" "$SOURCE_TRAIN_COUNT" "$JAVA_TRAIN_COUNT" \
    "$PYTHON_TRAIN_COUNT" "$SOURCE_VAL_COUNT" "$JAVA_VAL_COUNT" \
    "$PYTHON_VAL_COUNT" "$FROZEN_MANIFEST_SHA256"
}

build_args() {
  ARGS=(
    --version "$VERSION" --root "$ROOT" --run-dir "$RUN_DIR" --log-file "$LOG_FILE"
    --prepared-run "$PREPARED_RUN" --stress-run "$STRESS_RUN" --stability-run "$STABILITY_RUN"
    --resume-checkpoint "$RESUME_CHECKPOINT"
    --model-id "$MODEL_ID" --revision "$REVISION" --model-type "$MODEL_TYPE"
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
    --source-prepared "$SOURCE_PREPARED" --m1-run "$M1_RUN"
    --m1-adapter-sha256 "$M1_ADAPTER_SHA256" --m1-summary-sha256 "$M1_SUMMARY_SHA256"
    --m1-adapter-manifest-sha256 "$M1_ADAPTER_MANIFEST_SHA256"
    --m1-source-manifest-sha256 "$M1_SOURCE_MANIFEST_SHA256"
    --source-prepared-train-sha256 "$SOURCE_PREPARED_TRAIN_SHA256"
    --source-prepared-val-sha256 "$SOURCE_PREPARED_VAL_SHA256"
    --m1-optimizer-steps "$M1_OPTIMIZER_STEPS"
    --frozen-train-count "$FROZEN_TRAIN_COUNT" --frozen-val-count "$FROZEN_VAL_COUNT"
    --dataset-train-count "$DATASET_TRAIN_COUNT" --dataset-val-count "$DATASET_VAL_COUNT"
    --source-train-count "$SOURCE_TRAIN_COUNT" --java-train-count "$JAVA_TRAIN_COUNT"
    --python-train-count "$PYTHON_TRAIN_COUNT"
    --source-val-count "$SOURCE_VAL_COUNT" --java-val-count "$JAVA_VAL_COUNT"
    --python-val-count "$PYTHON_VAL_COUNT"
    --frozen-val-quota-policy "$FROZEN_VAL_QUOTA_POLICY"
    --pending-label-review-policy "$PENDING_LABEL_REVIEW_POLICY"
    --max-seq-length "$MAX_SEQ_LENGTH" --audit-progress-every "$AUDIT_PROGRESS_EVERY"
    --audit-failure-examples "$AUDIT_FAILURE_EXAMPLES"
    --seed "$SEED" --answer-yes "$ANSWER_YES" --answer-no "$ANSWER_NO"
    --system-text "$SYSTEM_TEXT" --user-prefix "$USER_PREFIX" --user-separator "$USER_SEPARATOR"
    --user-suffix "$USER_SUFFIX" --template-date "$TEMPLATE_DATE" --target-modules "$TARGET_MODULES"
    --min-target-modules "$MIN_TARGET_MODULES" --expected-trainable-parameters "$EXPECTED_TRAINABLE_PARAMETERS"
    --lora-rank "$LORA_RANK" --lora-alpha "$LORA_ALPHA" --lora-dropout "$LORA_DROPOUT"
    --lora-bias "$LORA_BIAS" --gradient-checkpointing "$GRADIENT_CHECKPOINTING"
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

printf '[G-02 %s] START MODE=%s CHICAGO=%s PIPELINE=%s LOG=%s\n' \
  "$VERSION" "$MODE" "$(at_chicago)" "$PIPELINE" "$LOG_FILE"
printf '[G-02 %s] FROZEN_F01C=%s M1_RUN=%s M1_ADAPTER_SHA256=%s\n' \
  "$VERSION" "$FROZEN_PREPARED" "$M1_RUN" "$M1_ADAPTER_SHA256"
printf '[G-02 %s] WARNING: F01C FROZEN DATASET STATUS=PREPARED_PENDING_LABEL_REVIEW; independent label review not claimed\n' "$VERSION"

run_stage() {
  local stage="$1" ordinal="$2" expected="$3" expected_steps="$4" 
  RUN_DIR="$PIPELINE/$ordinal-$stage"
  [[ ! -e "$RUN_DIR" ]] || { echo 'Refusing to overwrite stage dir' >&2; return 2; }
  mkdir "$RUN_DIR"
  build_args
  step "${stage}_engine" "$PYTHON_BIN" -u "$PYFILE" "$stage" "${ARGS[@]}"
  if [[ "$stage" == 'prepare' ]]; then
    EXPECTED_FINGERPRINT="$("$PYTHON_BIN" -c 'import json,sys;print(json.load(open(sys.argv[1]))["config_fingerprint"])' "$RUN_DIR/prepared/manifest.json")"
    PREPARED_RUN="$RUN_DIR"
  fi
  step "${stage}_proof" "$PYTHON_BIN" -u "$AUDITOR" stage \
    --pipeline "$PIPELINE" --stage-run "$RUN_DIR" --status "$expected" \
    --prepared-run "$PREPARED_RUN" --expected-fingerprint "$EXPECTED_FINGERPRINT" \
    --m1-adapter-sha256 "$M1_ADAPTER_SHA256" --log-file "$LOG_FILE" --step "$stage" \
    --steps "$expected_steps" --exit-code 0 \
    --max-gpu-gib "$MAX_GPU_GIB" --min-headroom-gib "$MIN_HEADROOM_GIB"
}

step FROZEN_VAL_QUOTA_PIN resolve_frozen_val_quotas
run_stage prepare 00 G02_V2_PREPARE_PASS_NOT_TRAINED 0
if [[ "$MODE" == '--train' ]]; then
  run_stage stress 01 G02_V2_STRESS_PASS_NOT_M2_TRAINING "$STRESS_STEPS"
  STRESS_RUN="$RUN_DIR"
  run_stage stability 02 G02_V2_STABILITY_PASS_NOT_M2_TRAINING "$STABILITY_STEPS"
  STABILITY_RUN="$RUN_DIR"
  run_stage train 03 G02_V2_M2_STANDARD_MIXED_TRAINING_COMPLETE "$MAX_STEPS"
fi
step FINAL_RESULT "$PYTHON_BIN" -u "$AUDITOR" finish \
  --pipeline "$PIPELINE" --stage-run "$RUN_DIR" --status '' \
  --prepared-run "$PREPARED_RUN" --expected-fingerprint "$EXPECTED_FINGERPRINT" \
  --m1-adapter-sha256 "$M1_ADAPTER_SHA256" --log-file "$LOG_FILE" \
  --step 'completed' --steps "$MAX_STEPS" --exit-code 0 \
  --max-gpu-gib "$MAX_GPU_GIB" --min-headroom-gib "$MIN_HEADROOM_GIB" \
  --operation="$MODE"
