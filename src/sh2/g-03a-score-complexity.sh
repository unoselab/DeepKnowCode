#!/usr/bin/env bash
# G-03a v1, GPT-OSS M3 D-04 Reduced-4D Complexity Scoring ONLY (no GPU/training).
# All experiment configuration lives in this standalone shell; Python receives required CLI.
set -Eeuo pipefail
export TZ=America/Chicago
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
PYTHON_BIN="$(command -v python)"
PYFILE="$ROOT/src/py2/g-03a-score-complexity.py"
VERSION="v1"
MODEL_ID="openai/gpt-oss-120b"
MODEL_REVISION="b5c939de8f754692c1647ca79fbf85e8c1e70f8a"
G02_PIPELINE="$ROOT/results/g-02/gpt-oss-120b/2026-1113-123203-1794594723-285189-27336-pipeline"
F01C_PREPARED="$ROOT/results/f-01c/codellama-7b/2026-1112-115525-prepare/prepared"
M1_ADAPTER="$ROOT/results/g-01d/gpt-oss-120b/2026-1113-004452-1794552292-275033-14249-pipeline/03-train/adapter/adapter_model.safetensors"
M1_ADAPTER_SHA256="0c9d913781a8a06342fef64113bfbb2ccd2d9d7dddd2b41d760915dbf8018cbf"
G02_PREPARED_MANIFEST_SHA256="61968384ce8855105a08658029c4e27825faf53d3fa93bb4211b3bca7438d292"
G02_PREPARED_TRAIN_SHA256="546142986ff39e08ccf38db6b6c12284c200f5bf20775c1b0b97f1a34fb63eeb"
G02_PREPARED_VALIDATION_SHA256="cf78092416d3ca8d039daf5c722490ec60d419232d17cc7a812f6a125d5b686c"
G02_STANDARD_ORDER_SHA256="b736ce03aacb6f3fa489ea1b90fb78c054e0a6f2a6e5d251bdccfb38182cb0e6"
G02_CONFIG_FINGERPRINT="262ef043bf069025f0db78f5d13b62af5082ec31a0d4662564c730ebcd374d03"
F01C_FROZEN_MANIFEST_SHA256="149b9439b8eeb2d5dbdf81931cd621b962e02066de0467ab330ce2a7fd933579"
F01C_FROZEN_TRAIN_SHA256="e2ec9d2cccc0adb0b73f81f60b3041b5a7f4fa519729f501c1cdc583c1e9b564"
F01C_FROZEN_VALIDATION_SHA256="4404f90262c1021917ce31f0c454f87a5a634a0875307b8e84bbd51abefc752a"
EXPECTED_TRAIN=19464
EXPECTED_VALIDATION=512
SOURCE_TRAIN=12000
JAVA_TRAIN=3732
PYTHON_TRAIN=3732
SOURCE_VALIDATION=412
JAVA_VALIDATION=60
PYTHON_VALIDATION=40
EXPECTED_STRUCTURAL_VARIANTS=8
MAX_SEQ_LENGTH=2048
GLOBAL_BATCH=24
OPTIMIZER_STEPS=811
SEED=2026
COMPONENT_POLICY="D04_VERIFIED_EFFECTIVE_CODE_COMPONENTS"
SCORE_POLICY="reduced-4d-externally-verified-v1"
SCORE_FIELD="reduced_4d_score"
SCORE_PROVENANCE="G03A_V1_RECOMPUTED_D04_EFFECTIVE_CODE_ON_G02_FROZEN"
SCORE_WEIGHTS=(0.45 0.25 0.20 0.10)
SCORE_TOLERANCE="0.000001"
D04_PARSER_VERSION_POLICY="d04-v2-tree-sitter-wrap"
D04_AST_FALLBACK_POLICY="d04-v5-corpus-label-median-explicit"
D04_NORMALIZATION_POLICY="d04-v2-corpus-train-p05-p95"
D04_SIMILARITY_POLICY="d04-v2-sequencematcher-autojunk-true"
D04_PERCENTILE_LOW="0.05"
D04_PERCENTILE_HIGH="0.95"
PARSER_TREE_SITTER_VERSION="0.25.2"
PARSER_JAVA_VERSION="0.23.5"
PARSER_PYTHON_VERSION="0.25.0"
MANUAL_LABEL_REVIEW_POLICY="PENDING_AND_NOT_INDEPENDENTLY_VERIFIED"
PROGRESS_INTERVAL=1000
RESULTS_DIR="$ROOT/results/g-03/gpt-oss-120b"
LOG_DIR="$ROOT/logs/g-03/gpt-oss-120b"
LOG_PREFIX="g-03a-v1"
LOG_TIMESTAMP_FORMAT='+%Y-%m%d-%H%M%S'
LOG_CAPTURE_STDOUT="true"
LOG_CAPTURE_STDERR="true"
LOG_MIRROR_TERMINAL="true"
LOG_NO_CLOBBER="true"

MODE="${1:---score}"
if (( $# )); then shift; fi
case "$MODE" in
  --score) [[ $# -eq 0 ]] || { echo 'Usage: --score' >&2; exit 2; }; PY_MODE=score ;;
  --verify) [[ $# -eq 1 ]] || { echo 'Usage: --verify ABSOLUTE_SCORE_RUN' >&2; exit 2; }; PY_MODE=verify; VERIFY_DIR="$1" ;;
  *) echo 'Usage: bash src/sh2/g-03a-score-complexity.sh --score | --verify ABSOLUTE_SCORE_RUN' >&2; exit 2 ;;
esac
mkdir -p "$RESULTS_DIR" "$LOG_DIR"
TS="$(TZ=America/Chicago date "$LOG_TIMESTAMP_FORMAT")"
RUN_UID="$(date +%s)-$$-$RANDOM"
LOG_FILE="$LOG_DIR/${LOG_PREFIX}-${PY_MODE}-${TS}-${RUN_UID}.log"
( set -o noclobber; : > "$LOG_FILE" ) || { echo 'LOG_COLLISION' >&2; exit 2; }
if [[ "$PY_MODE" == score ]]; then
  RUN_DIR="$RESULTS_DIR/${TS}-${RUN_UID}-g03a-score"
else
  [[ -d "$VERIFY_DIR" ]] || { echo "Missing score run: $VERIFY_DIR" >&2; exit 2; }
  RUN_DIR="$(cd "$VERIFY_DIR" && pwd -P)"
fi
[[ "$LOG_CAPTURE_STDOUT" == true && "$LOG_CAPTURE_STDERR" == true && "$LOG_MIRROR_TERMINAL" == true && "$LOG_NO_CLOBBER" == true ]] || {
  echo 'Cannot disable mandatory automatic log recording' >&2; exit 2;
}
exec > >(tee -a "$LOG_FILE") 2>&1
START_EPOCH="$(date +%s)"
LAST_STEP="INIT"
chi() { TZ=America/Chicago date --iso-8601=seconds; }
log() { printf '[G-03a %s] %s\n' "$VERSION" "$*"; }
finish() {
  local rc=$?
  trap - EXIT
  if (( rc == 0 )); then
    log "FINAL_STATUS=PASS EXIT=0 STEP=$LAST_STEP CHICAGO=$(chi) ELAPSED_SECONDS=$(($(date +%s)-START_EPOCH)) RUN=$RUN_DIR LOG=$LOG_FILE"
  else
    log "FINAL_STATUS=FAIL EXIT=$rc STEP=$LAST_STEP CHICAGO=$(chi) ELAPSED_SECONDS=$(($(date +%s)-START_EPOCH)) RUN=$RUN_DIR LOG=$LOG_FILE"
  fi
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
run_scoring_engine() {
  "$PYTHON_BIN" -u "$PYFILE" \
  --mode "$PY_MODE" --version "$VERSION" --root "$ROOT" \
  --g02-pipeline "$G02_PIPELINE" --f01c-prepared "$F01C_PREPARED" \
  --m1-adapter "$M1_ADAPTER" --run-dir "$RUN_DIR" --log-file "$LOG_FILE" \
  --model-id "$MODEL_ID" --model-revision "$MODEL_REVISION" \
  --g02-prepared-manifest-sha256 "$G02_PREPARED_MANIFEST_SHA256" \
  --g02-train-sha256 "$G02_PREPARED_TRAIN_SHA256" \
  --g02-validation-sha256 "$G02_PREPARED_VALIDATION_SHA256" \
  --g02-standard-order-sha256 "$G02_STANDARD_ORDER_SHA256" \
  --g02-config-fingerprint "$G02_CONFIG_FINGERPRINT" \
  --f01c-manifest-sha256 "$F01C_FROZEN_MANIFEST_SHA256" \
  --f01c-train-sha256 "$F01C_FROZEN_TRAIN_SHA256" \
  --f01c-validation-sha256 "$F01C_FROZEN_VALIDATION_SHA256" \
  --m1-adapter-sha256 "$M1_ADAPTER_SHA256" \
  --component-policy "$COMPONENT_POLICY" --score-policy "$SCORE_POLICY" \
  --score-field "$SCORE_FIELD" --score-provenance "$SCORE_PROVENANCE" \
  --ast-fallback-policy "$D04_AST_FALLBACK_POLICY" \
  --parser-version-policy "$D04_PARSER_VERSION_POLICY" \
  --normalization-policy "$D04_NORMALIZATION_POLICY" \
  --similarity-policy "$D04_SIMILARITY_POLICY" \
  --tree-sitter-version "$PARSER_TREE_SITTER_VERSION" \
  --tree-sitter-java-version "$PARSER_JAVA_VERSION" \
  --tree-sitter-python-version "$PARSER_PYTHON_VERSION" \
  --manual-label-review-policy "$MANUAL_LABEL_REVIEW_POLICY" \
  --expected-train "$EXPECTED_TRAIN" --expected-validation "$EXPECTED_VALIDATION" \
  --expected-variants "$EXPECTED_STRUCTURAL_VARIANTS" \
  --source-train "$SOURCE_TRAIN" --java-train "$JAVA_TRAIN" --python-train "$PYTHON_TRAIN" \
  --source-validation "$SOURCE_VALIDATION" --java-validation "$JAVA_VALIDATION" \
  --python-validation "$PYTHON_VALIDATION" \
  --max-seq-length "$MAX_SEQ_LENGTH" --progress-interval "$PROGRESS_INTERVAL" \
  --global-batch "$GLOBAL_BATCH" --optimizer-steps "$OPTIMIZER_STEPS" --seed "$SEED" \
  --percentile-low "$D04_PERCENTILE_LOW" --percentile-high "$D04_PERCENTILE_HIGH" \
  --score-tolerance "$SCORE_TOLERANCE" --score-weights "${SCORE_WEIGHTS[@]}"
}
main() {
  log "START MODE=$MODE CHICAGO=$(chi) RUN=$RUN_DIR LOG=$LOG_FILE"
  log "MODEL=$MODEL_ID REVISION=$MODEL_REVISION M1_SHA256=$M1_ADAPTER_SHA256"
  log "G02_FROZEN_TRAIN_SHA256=$G02_PREPARED_TRAIN_SHA256 F01C_FROZEN_TRAIN_SHA256=$F01C_FROZEN_TRAIN_SHA256"
  log "SCORE_POLICY=$SCORE_POLICY WEIGHTS=${SCORE_WEIGHTS[*]} REVIEW_STATUS=$MANUAL_LABEL_REVIEW_POLICY"
  step PYTHON_ENGINE_CHECK test -f "$PYFILE" || return $?
  step READ_ONLY_SCORE_AND_AUDIT run_scoring_engine || return $?
}
main
