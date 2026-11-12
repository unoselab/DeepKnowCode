#!/usr/bin/env bash
# F-01e v1 standalone Difficulty Score ingestion and Curriculum Order Audit.
# Run from repository root:
#   bash src/sh2/f-01e-score-curriculum.sh --inspect
#   F01E_SCORE_FILE=/absolute/path/to/verified_codellama_reduced4d.jsonl \
#   F01E_SCORE_SHA256=<sha256> F01E_SCORE_PROVENANCE=<evidence-id> \
#     bash src/sh2/f-01e-score-curriculum.sh --prepare
#   F01E_SCORE_FILE=... F01E_SCORE_SHA256=... F01E_SCORE_PROVENANCE=... \
#     bash src/sh2/f-01e-score-curriculum.sh --audit /absolute/path/to/prepare-run
# No GPU training. No previous wrapper invocation. No silent defaults in Python.
set -Eeuo pipefail
export TZ=America/Chicago
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
PYTHON_BIN="${F01E_PYTHON_BIN:-python}"
PYFILE="$ROOT/src/py2/f-01e-score-curriculum.py"
MODE="${1:---inspect}"
if (( $# )); then shift; fi
case "$MODE" in
  --inspect|--prepare) [[ $# -eq 0 ]] || { echo 'Unexpected arguments' >&2; exit 2; };;
  --audit) [[ $# -eq 1 ]] || { echo 'Usage: --audit /absolute/run' >&2; exit 2; }; AUDIT_RUN="$1";;
  *) echo 'Usage: --inspect | --prepare | --audit RUN_DIR' >&2; exit 2;;
esac
MODEL_ID="meta-llama/CodeLlama-7b-hf"
MODEL_REVISION="b462c3c99b077d341db691ec780a33156f3c1472"
MODEL_MODE="codellama-7b"
SEED=2026
GPU_IDS="0,1,2"
WORLD_SIZE=3
PRECISION="bf16"
PER_DEVICE_BATCH=1
GRADIENT_ACCUMULATION=8
GLOBAL_BATCH=24
MAX_SEQ_LENGTH=2048
OPTIMIZER_STEPS=811
TRAIN_COUNT=19464
VALIDATION_COUNT=512
SOURCE_COUNT=12000
JAVA_TARGET_COUNT=3732
PYTHON_TARGET_COUNT=3732
PHASE_NAMES=(Foundation Adaptation Specialization)
PHASE_STEPS=(270 270 271)
PHASE_SOURCE=(5184 4016 2800)
PHASE_JAVA=(648 1232 1852)
PHASE_PYTHON=(648 1232 1852)
FROZEN_PREPARED_RUN="$ROOT/results/f-01c/codellama-7b/2026-1112-115525-prepare"
FROZEN_TRAIN_SHA256="e2ec9d2cccc0adb0b73f81f60b3041b5a7f4fa519729f501c1cdc583c1e9b564"
FROZEN_VALIDATION_SHA256="4404f90262c1021917ce31f0c454f87a5a634a0875307b8e84bbd51abefc752a"
FROZEN_MANIFEST_SHA256="149b9439b8eeb2d5dbdf81931cd621b962e02066de0467ab330ce2a7fd933579"
M1_ADAPTER="$ROOT/results/f-01b/codellama-7b/2026-1111-232648-train/adapter/adapter_model.safetensors"
M1_SHA256="b8f057ca2e413402738ab7484c0b0108346f4d1c991f00e80ec21508514c78b4"
# The F-01c dataset stores IDs and token counts, NOT original code/AST.
# Explicit score file is mandatory for --prepare/--audit; never synthesize scores from IDs/tokens.
SCORE_FILE="${F01E_SCORE_FILE:-$ROOT/results/f-01e/verified_codellama_reduced4d.jsonl}"
SCORE_SHA256="${F01E_SCORE_SHA256:-}"
SCORE_POLICY="reduced-4d-externally-verified-v1"
SCORE_FIELD="reduced_4d_score"
SCORE_COMPONENT_FIELDS=(d_sim d_cf d_len d_df)
SCORE_WEIGHTS=(0.45 0.25 0.20 0.10)
SCORE_TOLERANCE="0.000001"
SCORE_PROVENANCE="${F01E_SCORE_PROVENANCE:-UNVERIFIED}"
if [[ "$MODE" == --inspect ]]; then
  SCORE_SHA256="$(printf '0%.0s' {1..64})"
  SCORE_PROVENANCE="INSPECT_ONLY"
else
  [[ -f "$SCORE_FILE" && -n "$SCORE_SHA256" && "$SCORE_PROVENANCE" != UNVERIFIED ]] || {
    echo 'BLOCKED: F01E_SCORE_FILE, F01E_SCORE_SHA256 and F01E_SCORE_PROVENANCE are required; no score will be inferred from IDs/tokens.' >&2
    exit 2
  }
fi
LOG_DIR="$ROOT/logs/f-01e/$MODEL_MODE"
RESULTS_DIR="$ROOT/results/f-01e/$MODEL_MODE"
LOG_PREFIX="f-01e-v1"
LOG_TIMESTAMP_FORMAT='+%Y-%m%d-%H%M%S'
mkdir -p "$LOG_DIR" "$RESULTS_DIR"
TS="$(TZ=America/Chicago date "$LOG_TIMESTAMP_FORMAT")"
RID="$(date +%s)-$$-$RANDOM"
LOG_FILE="$LOG_DIR/$LOG_PREFIX-${MODE#--}-$TS-$RID.log"
( set -o noclobber; : > "$LOG_FILE" ) || { echo 'LOG_COLLISION' >&2; exit 2; }
if [[ "$MODE" == --audit ]]; then
  RUN_DIR="$(cd "$AUDIT_RUN" && pwd -P)"
else
  RUN_DIR="$RESULTS_DIR/$TS-$RID-${MODE#--}"
fi
START_EPOCH="$(date +%s)"
finish() {
  local rc=$?
  trap - EXIT
  printf '[F-01e v1] FINAL STATUS=%s EXIT=%s CHICAGO=%s ELAPSED_SECONDS=%s RUN=%s LOG=%s\n' \
    "$([[ "$rc" -eq 0 ]] && echo PASS || echo FAIL)" "$rc" "$(TZ=America/Chicago date -Is)" \
    "$(( $(date +%s) - START_EPOCH ))" "$RUN_DIR" "$LOG_FILE" | tee -a "$LOG_FILE"
  exit "$rc"
}
trap finish EXIT
ARGS=(
  --root "$ROOT" --run-dir "$RUN_DIR" --prepared-run "$FROZEN_PREPARED_RUN"
  --m1-adapter "$M1_ADAPTER" --log-file "$LOG_FILE"
  --train-sha256 "$FROZEN_TRAIN_SHA256" --validation-sha256 "$FROZEN_VALIDATION_SHA256" --manifest-sha256 "$FROZEN_MANIFEST_SHA256"
  --m1-sha256 "$M1_SHA256" --model-id "$MODEL_ID" --model-revision "$MODEL_REVISION" --model-mode "$MODEL_MODE"
  --seed "$SEED" --gpu-ids "$GPU_IDS" --world-size "$WORLD_SIZE" --precision "$PRECISION"
  --per-device-batch "$PER_DEVICE_BATCH" --gradient-accumulation "$GRADIENT_ACCUMULATION" --global-batch "$GLOBAL_BATCH"
  --max-seq-length "$MAX_SEQ_LENGTH" --optimizer-steps "$OPTIMIZER_STEPS"
  --train-count "$TRAIN_COUNT" --validation-count "$VALIDATION_COUNT"
  --source-count "$SOURCE_COUNT" --java-target-count "$JAVA_TARGET_COUNT" --python-target-count "$PYTHON_TARGET_COUNT"
  --phase-names "${PHASE_NAMES[@]}" --phase-steps "${PHASE_STEPS[@]}"
  --phase-source "${PHASE_SOURCE[@]}" --phase-java "${PHASE_JAVA[@]}" --phase-python "${PHASE_PYTHON[@]}"
  --score-file "$SCORE_FILE" --score-sha256 "$SCORE_SHA256" --score-policy "$SCORE_POLICY"
  --score-field "$SCORE_FIELD" --score-provenance "$SCORE_PROVENANCE"
  --score-component-fields "${SCORE_COMPONENT_FIELDS[@]}" --score-weights "${SCORE_WEIGHTS[@]}"
  --score-tolerance "$SCORE_TOLERANCE"
  --log-prefix "$LOG_PREFIX" --log-timestamp-format "$LOG_TIMESTAMP_FORMAT"
)
{
  printf '[F-01e v1] START MODE=%s CHICAGO=%s RUN=%s LOG=%s\n' "$MODE" "$(TZ=America/Chicago date -Is)" "$RUN_DIR" "$LOG_FILE"
  printf '[F-01e v1] STEP=FROZEN_INPUT_AND_SCORE_AUDIT START CHICAGO=%s\n' "$(TZ=America/Chicago date -Is)"
  "$PYTHON_BIN" -u "$PYFILE" "${MODE#--}" "${ARGS[@]}"
  printf '[F-01e v1] STEP=FROZEN_INPUT_AND_SCORE_AUDIT PASS CHICAGO=%s\n' "$(TZ=America/Chicago date -Is)"
} 2>&1 | tee -a "$LOG_FILE"
