#!/usr/bin/env bash
# F-01d v1 — Curriculum preparation/audit only; NO GPU training in v1.
# Configuration is entirely in Shell; Python has no experimental defaults.
# Reproduce: bash src/sh2/f-01d-train-curriculum-lora.sh --prepare
# Audit:     bash src/sh2/f-01d-train-curriculum-lora.sh --audit /absolute/run/dir
set -Eeuo pipefail
export TZ=America/Chicago
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
PYTHON_BIN="${F01D_PYTHON_BIN:-python}"
PYFILE="$ROOT/src/py2/f-01d-train-curriculum-lora.py"
MODE="${1:---prepare}"
if (( $# )); then shift; fi
case "$MODE" in
  --prepare) [[ $# -eq 0 ]] || { echo 'Usage: --prepare' >&2; exit 2; } ;;
  --audit) [[ $# -eq 1 ]] || { echo 'Usage: --audit /absolute/prepared-run' >&2; exit 2; }; AUDIT_RUN="$1" ;;
  --smoke|--train) echo 'BLOCKED: F-01d v1 validates curriculum feasibility only; no approved difficulty scoring or GPU order integration yet' >&2; exit 2 ;;
  *) echo 'Usage: --prepare | --audit RUN_DIR' >&2; exit 2 ;;
esac
# FROZEN MODEL AND EXPERIMENT PARAMETERS
MODEL_ID="meta-llama/CodeLlama-7b-hf"
MODEL_REVISION="b462c3c99b077d341db691ec780a33156f3c1472"
MODEL_MODE="codellama-7b"
SEED=2026
GPU_IDS="0,1,2"
WORLD_SIZE=3
PRECISION="bf16"
MAX_SEQ_LENGTH=2048
GLOBAL_BATCH=24
OPTIMIZER_STEPS=811
TRAIN_COUNT=19464
VALIDATION_COUNT=512
SOURCE_COUNT=12000
JAVA_TARGET_COUNT=3732
PYTHON_TARGET_COUNT=3732
# Proposed D-04 three-phase plan, not yet a finalized difficulty-order policy.
PHASE_NAMES=(Foundation Adaptation Specialization)
PHASE_STEPS=(270 270 271)
PHASE_SOURCE=(5184 4016 2800)
PHASE_TARGET=(1296 2464 3704)
DIFFICULTY_POLICY="PENDING_D04_REDUCED_4D_SCORE_AUDIT"
SCORE_INPUT_POLICY="required_before_gpu_training"
FROZEN_M2_RUN="$ROOT/results/f-01c/codellama-7b/2026-1112-115525-prepare"
FROZEN_TRAIN_SHA256="e2ec9d2cccc0adb0b73f81f60b3041b5a7f4fa519729f501c1cdc583c1e9b564"
FROZEN_VAL_SHA256="4404f90262c1021917ce31f0c454f87a5a634a0875307b8e84bbd51abefc752a"
FROZEN_MANIFEST_SHA256="149b9439b8eeb2d5dbdf81931cd621b962e02066de0467ab330ce2a7fd933579"
M1_ADAPTER="$ROOT/results/f-01b/codellama-7b/2026-1111-232648-train/adapter/adapter_model.safetensors"
M1_ADAPTER_SHA256="b8f057ca2e413402738ab7484c0b0108346f4d1c991f00e80ec21508514c78b4"
# LOGGING
LOG_DIR="$ROOT/logs/f-01d/$MODEL_MODE"
RESULTS_DIR="$ROOT/results/f-01d/$MODEL_MODE"
LOG_PREFIX="f-01d-v1"
LOG_TIMESTAMP_FORMAT='+%Y-%m%d-%H%M%S'
mkdir -p "$LOG_DIR" "$RESULTS_DIR"
TS="$(TZ=America/Chicago date "$LOG_TIMESTAMP_FORMAT")"
RUN_ID="$(date +%s)-$$-${RANDOM}"
LOG_FILE="$LOG_DIR/$LOG_PREFIX-${MODE#--}-$TS-$RUN_ID.log"
( set -o noclobber; : > "$LOG_FILE" ) || { echo 'Existing log collision' >&2; exit 2; }
if [[ "$MODE" == --prepare ]]; then
  RUN_DIR="$RESULTS_DIR/$TS-$RUN_ID-curriculum-prepare"
else
  RUN_DIR="$(cd "$AUDIT_RUN" && pwd -P)"
fi
START_EPOCH="$(date +%s)"
finish() {
  local rc=$?
  trap - EXIT
  printf '[F-01d v1] FINAL STATUS=%s EXIT=%s CHICAGO=%s ELAPSED_SECONDS=%s RUN=%s LOG=%s\n' \
    "$([[ $rc -eq 0 ]] && echo PASS || echo FAIL)" "$rc" "$(TZ=America/Chicago date -Is)" "$(( $(date +%s) - START_EPOCH ))" "$RUN_DIR" "$LOG_FILE" | tee -a "$LOG_FILE"
  exit "$rc"
}
trap finish EXIT
ARGS=(
  --root "$ROOT" --run-dir "$RUN_DIR" --prepared-run "$FROZEN_M2_RUN"
  --train-sha256 "$FROZEN_TRAIN_SHA256" --validation-sha256 "$FROZEN_VAL_SHA256" --manifest-sha256 "$FROZEN_MANIFEST_SHA256"
  --m1-adapter "$M1_ADAPTER" --m1-sha256 "$M1_ADAPTER_SHA256"
  --model-id "$MODEL_ID" --model-revision "$MODEL_REVISION" --model-mode "$MODEL_MODE"
  --seed "$SEED" --gpu-ids "$GPU_IDS" --world-size "$WORLD_SIZE" --precision "$PRECISION"
  --max-seq-length "$MAX_SEQ_LENGTH" --global-batch "$GLOBAL_BATCH" --optimizer-steps "$OPTIMIZER_STEPS"
  --train-count "$TRAIN_COUNT" --validation-count "$VALIDATION_COUNT"
  --source-count "$SOURCE_COUNT" --java-target-count "$JAVA_TARGET_COUNT" --python-target-count "$PYTHON_TARGET_COUNT"
  --phase-names "${PHASE_NAMES[@]}" --phase-steps "${PHASE_STEPS[@]}"
  --phase-source "${PHASE_SOURCE[@]}" --phase-target "${PHASE_TARGET[@]}"
  --difficulty-policy "$DIFFICULTY_POLICY" --score-input-policy "$SCORE_INPUT_POLICY"
  --log-file "$LOG_FILE" --log-prefix "$LOG_PREFIX" --log-timestamp-format "$LOG_TIMESTAMP_FORMAT"
)
{
  printf '[F-01d v1] START MODE=%s CHICAGO=%s RUN=%s LOG=%s\n' "$MODE" "$(TZ=America/Chicago date -Is)" "$RUN_DIR" "$LOG_FILE"
  printf '[F-01d v1] STEP=HASH_AND_COUNTS START CHICAGO=%s\n' "$(TZ=America/Chicago date -Is)"
  "$PYTHON_BIN" -u "$PYFILE" "${MODE#--}" "${ARGS[@]}"
  printf '[F-01d v1] STEP=HASH_AND_COUNTS PASS CHICAGO=%s\n' "$(TZ=America/Chicago date -Is)"
} 2>&1 | tee -a "$LOG_FILE"
