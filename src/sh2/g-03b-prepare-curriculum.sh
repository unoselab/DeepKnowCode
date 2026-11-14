#!/usr/bin/env bash
# G-03b v1 -- CPU-only, read-only source inputs; generates AUDIT-ONLY M3 curriculum order.
# All experiment parameters are explicitly specified here; no other shell wrappers invoked.
set -Eeuo pipefail
export TZ=America/Chicago
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
PYTHON_BIN="$(command -v python)"
PYFILE="$ROOT/src/py2/g-03b-prepare-curriculum.py"
VERSION="v1"
MODEL_ID="openai/gpt-oss-120b"
MODEL_REVISION="b5c939de8f754692c1647ca79fbf85e8c1e70f8a"
MODEL_MODE="gpt-oss-120b"
G03A_SCORE_RUN="$ROOT/results/g-03/gpt-oss-120b/2026-1113-171543-1794611743-297518-969-g03a-score"
G02_PIPELINE="$ROOT/results/g-02/gpt-oss-120b/2026-1113-123203-1794594723-285189-27336-pipeline"
F01C_PREPARED="$ROOT/results/f-01c/codellama-7b/2026-1112-115525-prepare/prepared"
M1_ADAPTER="$ROOT/results/g-01d/gpt-oss-120b/2026-1113-004452-1794552292-275033-14249-pipeline/03-train/adapter/adapter_model.safetensors"
G03A_SCORE_MANIFEST_SHA256="fdc953fb5a4bf73a34bbe3cc1298f4cc520acf658d1bbc814f564df62388c78d"
G03A_SCORES_SHA256="ea2d56a9fb87a60831efd008edeb240966cf25e39542919a4093c532aeb15312"
G03A_COMPONENTS_SHA256="c07f7cdb12899701825325f179cb128d287cb10c17489edbeaa26366c7300b91"
G03A_COMPONENT_AUDIT_SHA256="27eebb3971958b5083bd78c8f9c41311b8a099ac92375dbd16a77dd683f8fb12"
G03A_SUMMARY_SHA256="5e9ea64dab9b7bff7ea140692b3400d9b077c08d3a5ab27b270cf89041d7965a"
G02_MANIFEST_SHA256="61968384ce8855105a08658029c4e27825faf53d3fa93bb4211b3bca7438d292"
G02_TRAIN_SHA256="546142986ff39e08ccf38db6b6c12284c200f5bf20775c1b0b97f1a34fb63eeb"
G02_VALIDATION_SHA256="cf78092416d3ca8d039daf5c722490ec60d419232d17cc7a812f6a125d5b686c"
G02_STANDARD_ORDER_SHA256="b736ce03aacb6f3fa489ea1b90fb78c054e0a6f2a6e5d251bdccfb38182cb0e6"
G02_FINGERPRINT="262ef043bf069025f0db78f5d13b62af5082ec31a0d4662564c730ebcd374d03"
EXPECTED_CURRICULUM_ORDER_SHA256="9e8ad5fdaaa1729aed4d75afe5911b5f617b2b6882b7a0d338c745e0c9d54766"
F01C_MANIFEST_SHA256="149b9439b8eeb2d5dbdf81931cd621b962e02066de0467ab330ce2a7fd933579"
F01C_TRAIN_SHA256="e2ec9d2cccc0adb0b73f81f60b3041b5a7f4fa519729f501c1cdc583c1e9b564"
F01C_VALIDATION_SHA256="4404f90262c1021917ce31f0c454f87a5a634a0875307b8e84bbd51abefc752a"
M1_ADAPTER_SHA256="0c9d913781a8a06342fef64113bfbb2ccd2d9d7dddd2b41d760915dbf8018cbf"
EXPECTED_TRAIN=19464
EXPECTED_VALIDATION=512
SOURCE_TRAIN=12000
JAVA_TRAIN=3732
PYTHON_TRAIN=3732
SOURCE_VALIDATION=412
JAVA_VALIDATION=60
PYTHON_VALIDATION=40
GLOBAL_BATCH=24
OPTIMIZER_STEPS=811
MAX_SEQ_LENGTH=2048
SEED=2026
EXPECTED_IMPUTED=5
EXPECTED_VARIANTS=8
SCORE_POLICY="reduced-4d-externally-verified-v1"
SCORE_PROVENANCE="G03A_V1_RECOMPUTED_D04_EFFECTIVE_CODE_ON_G02_FROZEN"
SCORE_FIELD="reduced_4d_score"
SCORE_WEIGHTS=(0.45 0.25 0.20 0.10)
SCORE_TOLERANCE="0.000001"
CURRICULUM_POLICY="three-phase-domain-balanced-easy-to-hard-v1"
TIE_POLICY="score-seeded-sha-row-v1"
REVIEW_POLICY="PENDING_AND_NOT_INDEPENDENTLY_VERIFIED"
TRAINING_APPROVED="false"
PHASE_NAMES=(Foundation Adaptation Specialization)
PHASE_STEPS=(270 270 271)
PHASE_SOURCE=(5184 4016 2800)
PHASE_TARGET=(1296 2464 3704)
RESULTS_DIR="$ROOT/results/g-03/$MODEL_MODE"
LOG_DIR="$ROOT/logs/g-03/$MODEL_MODE"
LOG_PREFIX="g-03b-v1"
LOG_TIMESTAMP_FORMAT='+%Y-%m%d-%H%M%S'
LOG_CAPTURE_STDOUT="true"
LOG_CAPTURE_STDERR="true"
LOG_MIRROR_TERMINAL="true"
LOG_NO_CLOBBER="true"
MODE="${1:---prepare}"
if (( $# )); then shift; fi
case "$MODE" in
  --prepare) [[ $# -eq 0 ]] || { echo 'Usage: --prepare' >&2; exit 2; }; PY_MODE=prepare ;;
  --verify) [[ $# -eq 1 ]] || { echo 'Usage: --verify RUN_DIR' >&2; exit 2; }; PY_MODE=verify; VERIFY_RUN="$1" ;;
  *) echo 'Usage: bash src/sh2/g-03b-prepare-curriculum.sh --prepare | --verify RUN_DIR' >&2; exit 2 ;;
esac
mkdir -p "$RESULTS_DIR" "$LOG_DIR"
STAMP="$(TZ=America/Chicago date "$LOG_TIMESTAMP_FORMAT")"
RUN_UID="$(date +%s)-$$-$RANDOM"
LOG_FILE="$LOG_DIR/${LOG_PREFIX}-${PY_MODE}-${STAMP}-${RUN_UID}.log"
( set -o noclobber; : > "$LOG_FILE" ) || { echo 'LOG_COLLISION' >&2; exit 2; }
if [[ "$PY_MODE" == prepare ]]; then
  RUN_DIR="$RESULTS_DIR/${STAMP}-${RUN_UID}-g03b-curriculum"
else
  [[ -d "$VERIFY_RUN" ]] || { echo "RUN_DIR_MISSING: $VERIFY_RUN" >&2; exit 2; }
  RUN_DIR="$(cd "$VERIFY_RUN" && pwd -P)"
fi
[[ "$LOG_CAPTURE_STDOUT" == true && "$LOG_CAPTURE_STDERR" == true &&
   "$LOG_MIRROR_TERMINAL" == true && "$LOG_NO_CLOBBER" == true ]] || {
  echo 'LOGGING_MUST_REMAIN_AUTOMATIC' >&2; exit 2;
}
exec > >(tee -a "$LOG_FILE") 2>&1
START_EPOCH="$(date +%s)"
LAST_STEP="INIT"
chi() { TZ=America/Chicago date --iso-8601=seconds; }
log() { printf '[G-03b %s] %s\n' "$VERSION" "$*"; }
finish() {
  local rc=$?
  trap - EXIT
  log "FINAL_STATUS=$([[ $rc -eq 0 ]] && echo PASS || echo FAIL) EXIT=$rc STEP=$LAST_STEP CHICAGO=$(chi) ELAPSED_SECONDS=$(($(date +%s)-START_EPOCH)) RUN=$RUN_DIR LOG=$LOG_FILE"
  exit "$rc"
}
trap finish EXIT
step() {
  local label="$1" begin=$SECONDS rc=0
  shift
  LAST_STEP="$label"
  log "STEP=$label START CHICAGO=$(chi)"
  "$@" || rc=$?
  if (( rc != 0 )); then
    log "STEP=$label FAIL EXIT=$rc ELAPSED_SECONDS=$((SECONDS-begin)) CHICAGO=$(chi)"
    return "$rc"
  fi
  log "STEP=$label PASS EXIT=0 ELAPSED_SECONDS=$((SECONDS-begin)) CHICAGO=$(chi)"
}
run_engine() {
  "$PYTHON_BIN" -u "$PYFILE" \
  --mode "$PY_MODE" --version "$VERSION" --root "$ROOT" --log-file "$LOG_FILE" --run-dir "$RUN_DIR" \
  --g03a-score-run "$G03A_SCORE_RUN" --g02-pipeline "$G02_PIPELINE" \
  --f01c-prepared "$F01C_PREPARED" --m1-adapter "$M1_ADAPTER" \
  --model-id "$MODEL_ID" --model-revision "$MODEL_REVISION" \
  --g03a-score-manifest-sha256 "$G03A_SCORE_MANIFEST_SHA256" \
  --g03a-scores-sha256 "$G03A_SCORES_SHA256" \
  --g03a-components-sha256 "$G03A_COMPONENTS_SHA256" \
  --g03a-component-audit-sha256 "$G03A_COMPONENT_AUDIT_SHA256" \
  --g03a-summary-sha256 "$G03A_SUMMARY_SHA256" \
  --g02-manifest-sha256 "$G02_MANIFEST_SHA256" \
  --g02-train-sha256 "$G02_TRAIN_SHA256" \
  --g02-validation-sha256 "$G02_VALIDATION_SHA256" \
  --g02-standard-order-sha256 "$G02_STANDARD_ORDER_SHA256" \
  --g02-fingerprint "$G02_FINGERPRINT" \
  --expected-curriculum-order-sha256 "$EXPECTED_CURRICULUM_ORDER_SHA256" \
  --f01c-manifest-sha256 "$F01C_MANIFEST_SHA256" \
  --f01c-train-sha256 "$F01C_TRAIN_SHA256" \
  --f01c-validation-sha256 "$F01C_VALIDATION_SHA256" \
  --m1-adapter-sha256 "$M1_ADAPTER_SHA256" \
  --score-policy "$SCORE_POLICY" --score-provenance "$SCORE_PROVENANCE" \
  --score-field "$SCORE_FIELD" --score-weights "${SCORE_WEIGHTS[@]}" \
  --score-tolerance "$SCORE_TOLERANCE" --curriculum-policy "$CURRICULUM_POLICY" \
  --tie-policy "$TIE_POLICY" --review-policy "$REVIEW_POLICY" \
  --training-approved "$TRAINING_APPROVED" \
  --expected-train "$EXPECTED_TRAIN" --expected-validation "$EXPECTED_VALIDATION" \
  --source-train "$SOURCE_TRAIN" --java-train "$JAVA_TRAIN" --python-train "$PYTHON_TRAIN" \
  --source-validation "$SOURCE_VALIDATION" --java-validation "$JAVA_VALIDATION" \
  --python-validation "$PYTHON_VALIDATION" \
  --global-batch "$GLOBAL_BATCH" --optimizer-steps "$OPTIMIZER_STEPS" \
  --max-seq-length "$MAX_SEQ_LENGTH" --seed "$SEED" \
  --expected-imputed "$EXPECTED_IMPUTED" --expected-variants "$EXPECTED_VARIANTS" \
  --phase-names "${PHASE_NAMES[@]}" --phase-steps "${PHASE_STEPS[@]}" \
  --phase-source "${PHASE_SOURCE[@]}" --phase-target "${PHASE_TARGET[@]}"
}
main() {
  log "START MODE=$MODE CHICAGO=$(chi) RUN=$RUN_DIR LOG=$LOG_FILE"
  log "MODEL=$MODEL_ID REVISION=$MODEL_REVISION M1_SHA256=$M1_ADAPTER_SHA256"
  log "G03A_SCORE_SHA256=$G03A_SCORES_SHA256 G02_FROZEN_TRAIN_SHA256=$G02_TRAIN_SHA256"
  log "PHASES=${PHASE_NAMES[*]} STEPS=${PHASE_STEPS[*]} SOURCE=${PHASE_SOURCE[*]} TARGET=${PHASE_TARGET[*]}"
  log "APPROVED_FOR_TRAINING=$TRAINING_APPROVED LABEL_REVIEW=$REVIEW_POLICY"
  step PYTHON_ENGINE_CHECK test -f "$PYFILE" || return $?
  step READ_ONLY_CURRICULUM_PREP_AND_AUDIT run_engine || return $?
}
main
