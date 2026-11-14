#!/usr/bin/env bash
# G-03c v4. Independent M3 curriculum from the SAME G01d M1 adapter as G02 M2.
# Frozen G02 data, G03a scores, G03b exact 811x24 order. No other shell wrapper invoked.
# --train is operator-authorized exploratory M3; pending review remains documented.
set -Eeuo pipefail
export TZ=America/Chicago
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
PYTHON_BIN="python"
PYFILE="$ROOT/src/py2/g-03c-train-curriculum-lora.py"
ENGINE="$ROOT/src/py2/g-03c-training-kernel.py"
VERSION="v4"
CORE_VERSION="v2"
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


# G-02 v2 completed frozen inputs, EXACTLY as used by G03a/G03b.
G02_PIPELINE="$ROOT/results/g-02/gpt-oss-120b/2026-1113-123203-1794594723-285189-27336-pipeline"
G03A_RUN="$ROOT/results/g-03/gpt-oss-120b/2026-1113-171543-1794611743-297518-969-g03a-score"
G03B_RUN="$ROOT/results/g-03/gpt-oss-120b/2026-1113-221148-1794629508-303275-31812-g03b-curriculum"
G03B_MANIFEST_SHA256="2182dce2a86821a02c2b989586737c022d4a9dd0363b8d2454921cccfd2c4f5c"
G03B_ORDER_SHA256="9e8ad5fdaaa1729aed4d75afe5911b5f617b2b6882b7a0d338c745e0c9d54766"
G03B_PHASE_AUDIT_SHA256="44df574d934465e2f13e20cd5b52b4603804cedc3eb31f9a09486a472ea15b54"
G03B_COMPARISON_SHA256="90bf3aadf4884c3ac121530e2ab7225b3c51d9c866653630058072e90e993a01"
G02_PREPARED_MANIFEST_SHA256="61968384ce8855105a08658029c4e27825faf53d3fa93bb4211b3bca7438d292"
G02_PREPARED_TRAIN_SHA256="546142986ff39e08ccf38db6b6c12284c200f5bf20775c1b0b97f1a34fb63eeb"
G02_PREPARED_VAL_SHA256="cf78092416d3ca8d039daf5c722490ec60d419232d17cc7a812f6a125d5b686c"
G02_STANDARD_ORDER_SHA256="b736ce03aacb6f3fa489ea1b90fb78c054e0a6f2a6e5d251bdccfb38182cb0e6"
G02_FINGERPRINT="262ef043bf069025f0db78f5d13b62af5082ec31a0d4662564c730ebcd374d03"
G03A_SCORES_SHA256="ea2d56a9fb87a60831efd008edeb240966cf25e39542919a4093c532aeb15312"
ENGINE_SHA256="0efe383ac83a8c4fe6c05ae79ee532fc38848fdc255c454cafefdb7e586f29d8"
# Shell owns all M3 curriculum parameters, not Python defaults.
CURRICULUM_POLICY="three-phase-domain-balanced-easy-to-hard-v1"
TRAINING_APPROVED="false"
MANUAL_LABEL_REVIEW_STATUS="PENDING_AND_NOT_INDEPENDENTLY_VERIFIED"
EXPECTED_IMPUTED=5
EXPECTED_VARIANTS=8
PHASE_NAMES=(Foundation Adaptation Specialization)
PHASE_STEPS=(270 270 271)
PHASE_SOURCE=(5184 4016 2800)
PHASE_TARGET=(1296 2464 3704)
# G-03b/G-02 validation quotas are SHA256 pinned; never derive guesses in Python.
SOURCE_VAL_COUNT=412
JAVA_VAL_COUNT=60
PYTHON_VAL_COUNT=40
# Logging: Chicago, unique, stdout+stderr captured via tee, no overwrites.
LOG_DIR="$ROOT/logs/g-03/gpt-oss-120b"
RESULTS_DIR="$ROOT/results/g-03/gpt-oss-120b"
LOG_PREFIX="g-03c-v4"
LOG_TIMESTAMP_FORMAT='+%Y-%m%d-%H%M%S'
LOG_TEE_OPTIONS='-a'
LOG_CAPTURE_STDOUT="true"
LOG_CAPTURE_STDERR="true"
LOG_PIPEFAIL="true"
[[ $# -ge 1 ]] || { echo 'Usage: --audit | --train | --verify M3_PIPELINE' >&2; exit 2; }
MODE="$1"; shift
case "$MODE" in
  --audit|--train) [[ $# -eq 0 ]] || { echo 'No positional overrides allowed' >&2; exit 2; } ;;
  --verify) [[ $# -eq 1 ]] || { echo 'Usage: --verify /absolute/previous-M3-pipeline' >&2; exit 2; }; VERIFY_SOURCE="$1" ;;
  *) echo 'Usage: --audit | --train | --verify M3_PIPELINE' >&2; exit 2 ;;
esac
[[ -f "$PYFILE" && -f "$ENGINE" ]] || { echo 'Missing G03c controller or vendored kernel' >&2;exit 2; }
[[ "${CONDA_DEFAULT_ENV:-}" == "$CONDA_ENV" ]] || { echo "Activate conda env: $CONDA_ENV" >&2;exit 2; }
if [[ "$(sha256sum "$ENGINE" | awk '{print $1}')" != "$ENGINE_SHA256" ]]; then
  echo 'G03C pinned G02 compatibility kernel SHA256 mismatch' >&2; exit 2
fi
export CUDA_VISIBLE_DEVICES="$DEVICE_IDS" TOKENIZERS_PARALLELISM=false
mkdir -p "$LOG_DIR" "$RESULTS_DIR"
STAMP="$(TZ=America/Chicago date "$LOG_TIMESTAMP_FORMAT")"
UNIQUE="$(date +%s)-$$-$RANDOM"
PIPELINE="$RESULTS_DIR/$STAMP-$UNIQUE-g03c-pipeline"
LOG_FILE="$LOG_DIR/$LOG_PREFIX-${MODE#--}-$STAMP-$UNIQUE.log"
( set -C; : > "$LOG_FILE" ) || { echo 'Refusing to overwrite existing log' >&2;exit 2; }
[[ ! -e "$PIPELINE" ]] || { echo 'Refusing to overwrite existing run' >&2;exit 2; }
mkdir "$PIPELINE"
exec > >(tee "$LOG_TEE_OPTIONS" "$LOG_FILE") 2>&1
START_SEC="$(date +%s)"
ACTIVE_STAGE="init"
RESEARCH_ACKNOWLEDGED="false"
if [[ "$MODE" == '--train' ]]; then RESEARCH_ACKNOWLEDGED="true"; fi
PREPARED_RUN="$G02_PIPELINE/00-prepare"
STRESS_RUN=''
STABILITY_RUN=''
RESUME_CHECKPOINT=''
RUN_DIR=''
AUDIT_RUN=''
TRAIN_RUN=''
at_chicago() { TZ=America/Chicago date -Is; }
finish() {
  local rc=$?
  trap - EXIT
  if [[ ! -f "$PIPELINE/pipeline_result.json" ]]; then
    "$PYTHON_BIN" - "$PIPELINE/pipeline_result.json" "$VERSION" "$MODE" "$rc" "$ACTIVE_STAGE" "$LOG_FILE" <<'PYRESULT' || true
import json, pathlib, sys
from datetime import datetime
from zoneinfo import ZoneInfo
p=pathlib.Path(sys.argv[1]);version,mode,rc,stage,log=sys.argv[2:]
data={'version':version,'mode':mode,'exit_code':int(rc),'status':'PASS' if int(rc)==0 else 'FAIL',
      'last_stage':stage,'log_file':log,'at_chicago':datetime.now(ZoneInfo('America/Chicago')).isoformat()}
with p.open('x') as out: json.dump(data,out,indent=2,sort_keys=True)
PYRESULT
  fi
  printf '[G-03c %s] FINAL_STATUS=%s EXIT=%s STEP=%s CHICAGO=%s DURATION_SECONDS=%s PIPELINE=%s LOG=%s\n' \
    "$VERSION" "$( [[ $rc -eq 0 ]] && echo PASS || echo FAIL )" "$rc" \
    "$ACTIVE_STAGE" "$(at_chicago)" "$(( $(date +%s) - START_SEC ))" "$PIPELINE" "$LOG_FILE"
  exit "$rc"
}
trap finish EXIT
step() {
  local name="$1" rc start
  shift
  ACTIVE_STAGE="$name"; start="$(date +%s)"
  printf '[G-03c %s] STEP=%s START CHICAGO=%s\n' "$VERSION" "$name" "$(at_chicago)"
  if "$@"; then
    printf '[G-03c %s] STEP=%s PASS ELAPSED_SECONDS=%s CHICAGO=%s\n' \
      "$VERSION" "$name" "$(( $(date +%s)-start ))" "$(at_chicago)"
  else
    rc=$?
    printf '[G-03c %s] STEP=%s FAIL EXIT=%s ELAPSED_SECONDS=%s CHICAGO=%s\n' \
      "$VERSION" "$name" "$rc" "$(( $(date +%s)-start ))" "$(at_chicago)"
    return "$rc"
  fi
}
build_args() {
  ARGS=(
    --version "$CORE_VERSION" --root "$ROOT" --run-dir "$RUN_DIR" --log-file "$LOG_FILE"
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

controller() {
  local stage="$1" core_mode="prepare"
  # Stage-specific G-02 compatibility inputs MUST NOT mutate caller state.
  # v2 cleared the global PREPARED_RUN during audit; stress then received "".
  # Keep these shell-owned values explicit for every stage (also for stand-alone verify).
  local PREPARED_RUN='' STRESS_RUN='' STABILITY_RUN=''
  case "$stage" in
    audit|verify)
      core_mode='prepare'
      ;;
    stress)
      core_mode='stress'
      PREPARED_RUN="$G02_PIPELINE/00-prepare"
      ;;
    stability)
      core_mode='stability'
      PREPARED_RUN="$G02_PIPELINE/00-prepare"
      STRESS_RUN="$PIPELINE/01-stress"
      ;;
    train)
      core_mode='train'
      PREPARED_RUN="$G02_PIPELINE/00-prepare"
      STRESS_RUN="$PIPELINE/01-stress"
      STABILITY_RUN="$PIPELINE/02-stability"
      ;;
    *) echo "Unsupported stage: $stage" >&2; return 2 ;;
  esac
  build_args
  "$PYTHON_BIN" -u "$PYFILE" "$stage" \
    --version "$VERSION" --pipeline "$PIPELINE" --stage-run "$RUN_DIR" \
    --model-id "$MODEL_ID" --model-revision "$REVISION" \
    --g03b-run "$G03B_RUN" --g02-pipeline "$G02_PIPELINE" --g03a-run "$G03A_RUN" \
    --audit-run "${AUDIT_RUN:--}" --stress-run "${STRESS_RUN:--}" \
    --stability-run "${STABILITY_RUN:--}" --train-run "${TRAIN_RUN:--}" \
    --engine-path "$ENGINE" --engine-sha256 "$ENGINE_SHA256" \
    --g03b-manifest-sha256 "$G03B_MANIFEST_SHA256" \
    --g03b-order-sha256 "$G03B_ORDER_SHA256" \
    --g03b-phase-audit-sha256 "$G03B_PHASE_AUDIT_SHA256" \
    --g03b-comparison-sha256 "$G03B_COMPARISON_SHA256" \
    --g02-prepared-manifest-sha256 "$G02_PREPARED_MANIFEST_SHA256" \
    --g02-train-sha256 "$G02_PREPARED_TRAIN_SHA256" \
    --g02-validation-sha256 "$G02_PREPARED_VAL_SHA256" \
    --g02-standard-order-sha256 "$G02_STANDARD_ORDER_SHA256" \
    --g02-fingerprint "$G02_FINGERPRINT" --g03a-score-sha256 "$G03A_SCORES_SHA256" \
    --m1-adapter-sha256 "$M1_ADAPTER_SHA256" \
    --research-acknowledged "$RESEARCH_ACKNOWLEDGED" \
    --training-approved "$TRAINING_APPROVED" \
    --label-review-status "$MANUAL_LABEL_REVIEW_STATUS" \
    --phase-policy "$CURRICULUM_POLICY" \
    --expected-imputed "$EXPECTED_IMPUTED" --expected-variants "$EXPECTED_VARIANTS" \
    --train-pairs "$DATASET_TRAIN_COUNT" --validation-pairs "$DATASET_VAL_COUNT" \
    --optimizer-steps "$MAX_STEPS" \
    --global-batch "$(( PER_DEVICE_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS ))" \
    --seed "$SEED" --phase-names "${PHASE_NAMES[@]}" \
    --phase-steps "${PHASE_STEPS[@]}" --phase-source "${PHASE_SOURCE[@]}" \
    --phase-target "${PHASE_TARGET[@]}" \
    --kernel-args "$core_mode" "${ARGS[@]}"
}
run_stage() {
  local name="$1" dir_name="$2"
  RUN_DIR="$PIPELINE/$dir_name-$name"
  mkdir "$RUN_DIR"
  step "$name" controller "$name"
  local proof=''
  case "$name" in
    audit) proof='audit_manifest.json'; AUDIT_RUN="$RUN_DIR" ;;
    stress) proof='g03c_stage_proof.json' ;;
    stability) proof='g03c_stage_proof.json' ;;
    train) proof='g03c_train_manifest.json'; TRAIN_RUN="$RUN_DIR" ;;
  esac
  [[ -n "$proof" && -s "$RUN_DIR/$proof" ]] || {
     echo "ERROR: $name returned without required proof file $proof" >&2; return 4;
  }
}
printf '[G-03c %s] START MODE=%s CHICAGO=%s PIPELINE=%s LOG=%s\n' \
   "$VERSION" "$MODE" "$(at_chicago)" "$PIPELINE" "$LOG_FILE"
printf '[G-03c %s] G03B_ORDER_SHA256=%s M1_SHA256=%s G02_FROZEN_SHA256=%s\n' \
   "$VERSION" "$G03B_ORDER_SHA256" "$M1_ADAPTER_SHA256" "$G02_PREPARED_TRAIN_SHA256"
printf '[G-03c %s] RESEARCH_ACKNOWLEDGED=%s TRAINING_APPROVED=%s LABEL_REVIEW=%s\n' \
   "$VERSION" "$RESEARCH_ACKNOWLEDGED" "$TRAINING_APPROVED" "$MANUAL_LABEL_REVIEW_STATUS"
if [[ "$MODE" == '--train' ]]; then
  printf '[G-03c %s] OPERATOR_ASSUMPTION=LABEL_REVIEW_ASSUMED_CONFIRMED; FROZEN_EVIDENCE_STATUS=%s; INDEPENDENT_REVIEW_NOT_VERIFIED_BY_THIS_RUN=true\n' \
    "$VERSION" "$MANUAL_LABEL_REVIEW_STATUS"
fi
if [[ "$MODE" == '--audit' ]]; then
  run_stage audit 00
elif [[ "$MODE" == '--train' ]]; then
  # Operator-authorized research training; preserve actual pending-review provenance.
  run_stage audit 00
  run_stage stress 01
  run_stage stability 02
  run_stage train 03
  RUN_DIR="$PIPELINE/04-verify"
  mkdir "$RUN_DIR"
  step verify controller verify
  [[ -s "$RUN_DIR/verification_result.json" ]] || { echo 'M3 independent verification proof missing' >&2; exit 4; }
elif [[ "$MODE" == '--verify' ]]; then
  [[ -d "$VERIFY_SOURCE/03-train" ]] || { echo 'Missing previous M3 train stage' >&2; exit 2; }
  TRAIN_RUN="$(cd "$VERIFY_SOURCE/03-train" && pwd -P)"
  RUN_DIR="$PIPELINE/00-verify"
  mkdir "$RUN_DIR"
  step verify controller verify
  [[ -s "$RUN_DIR/verification_result.json" ]] || { echo 'M3 independent verification proof missing' >&2; exit 4; }
fi
