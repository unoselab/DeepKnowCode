#!/usr/bin/env bash
# =============================================================================
# DeepKnowCode J-03d v8 - Positive Clones + H-07/H-08 AST Type Audit
# =============================================================================
# Run from the DeepKnowCode project root in this order:
#
# 1. Preflight verification (47 J-03c AFTER subjects)
# bash src/sh3/j-03d-positive-clones.sh --inspect
#
# 2. Smoke test: apache__ant
# bash src/sh3/j-03d-positive-clones.sh --smoke
#
# 3. Full 47-repository export
# bash src/sh3/j-03d-positive-clones.sh --all
#
# 4. Final verification
# bash src/sh3/j-03d-positive-clones.sh --verify
#
# 5. AST clone-type smoke test (apache__ant, read-only)
# bash src/sh3/j-03d-positive-clones.sh --clonetypes-smoke
#
# 6. Full H-07/H-08 Tree-sitter clone-type candidate census
# bash src/sh3/j-03d-positive-clones.sh --clonetypes
#
# 7. Every unique Type 1/2/3 positive code-content pair (native subject ratios)
# bash src/sh3/j-03d-positive-clones.sh --positiveclone
# bash src/sh3/j-03d-positive-clones.sh --positiveclone-verify
#
# 8. Legacy v6 class-balanced positives, relocated into sameratio-type/
# bash src/sh3/j-03d-positive-clones.sh --positiveclone-sameratio-type
# bash src/sh3/j-03d-positive-clones.sh --positiveclone-sameratio-type-verify
#
# Optional single subject:
# bash src/sh3/j-03d-positive-clones.sh --subject apache__camel
#
# Authoritative clone pair INPUT: data/clone-detection/2after-preproc/
# Java code SOURCE: data/repositories/java-repo-preproc/
# Positive pair OUTPUT: data/clone-detection/3pos-clones/
# Never read 1before-preproc or overwrite existing verified output.
# Shell owns all parameters and America/Chicago step-level logging.
# =============================================================================
set -Eeuo pipefail

# --------------------------- Experiment configuration -----------------------
EXPERIMENT="j-03d"
VERSION="v8"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
EXPECTED_INPUT_RELATIVE="data/clone-detection/2after-preproc"
EXPECTED_OUTPUT_RELATIVE="data/clone-detection/3pos-clones"
INPUT_ROOT="$ROOT/$EXPECTED_INPUT_RELATIVE"
SOURCE_ROOT="$ROOT/data/repositories/java-repo-preproc"
OUTPUT_ROOT="$ROOT/$EXPECTED_OUTPUT_RELATIVE"
RESULT_ROOT="$ROOT/results/j-03d"
PYTHON_BIN="python3"
PYTHON_SCRIPT="$ROOT/src/py3/j-03d-positive-clones.py"
EXPECTED_SUBJECTS=47
EXPECTED_TOTAL_PAIRS=3790008
SMOKE_SUBJECT="apache__ant"
EXPECTED_INPUT_EXPERIMENT="j-03c"
EXPECTED_INPUT_VERSION="v1"
EXPECTED_INPUT_PHASE="after"
EXPECTED_INPUT_STATUS="ARCHIVED_VERIFIED"
EXPECTED_PREPROC_EXPERIMENT="j-03b"
EXPECTED_PREPROC_VERSION="v2"
EXPECTED_PREPROC_STATUS="PREPARED"
EXPECTED_NICAD_CONFIG="default"
EXPECTED_NICAD_THRESHOLD="0.30"
EXPECTED_DETECTOR="NiCad7/functions/java"
EXPECTED_LANGUAGE="java"
POSITIVE_LABEL=1
CODE_ENCODING="utf-8"
CODE_ERRORS="surrogateescape"
PROGRESS_EVERY_FRAGMENTS=10000
PROGRESS_EVERY_PAIRS=50000
# Accept immutable v2/v3/v4 datasets; only missing exports will carry v5 manifests.
ACCEPTED_OUTPUT_VERSIONS="v2,v3,v4,v5,v6,v7,v8"
# Legacy v3 similarity-only parameters are retained for compatibility with the
# immutable v2/v3 positive dataset and their effective configuration.
CLONETYPE_T1_LABEL="T1_EXACT_PREPROC_PROXY"
CLONETYPE_T2_LABEL="T2_NICAD_100_PROXY"
CLONETYPE_T3_LABEL="T3_NICAD_NEARMISS_PROXY"
CLONETYPE_T4_LABEL="T4_UNDETERMINED"
CLONETYPE_MIN_SIMILARITY="70"
CLONETYPE_MAX_SIMILARITY="100"
CLONETYPE_REPORT_NAME="summary-clonetype.md"
CLONETYPE_PROGRESS_EVERY_PAIRS=500000
# New H-07 / H-08 AST classification; ALL parameters specified in Shell.
H07_PYTHON="$ROOT/src/py2/h-07-audit-java-tree-sitter.py"
H08_PYTHON="$ROOT/src/py2/h-08-analyze-java-clone-types.py"
AST_IMPL="$ROOT/src/py3/j-03d-ast-clonetypes.py"
AST_PARSER_VERSION="0.25.2"
AST_GRAMMAR_VERSION="0.23.5"
AST_GRAM_SIZE=5
AST_PRIMARY_THRESHOLD="0.82"
AST_THRESHOLDS="0.70,0.75,0.80,0.82,0.85,0.90,0.95"
AST_MAX_COMPARISON_CELLS=4000000
AST_PROGRESS_FRAGMENTS=10000
AST_PROGRESS_PAIRS=50000
AST_FRAGMENT_CACHE_SIZE=4096
AST_SCORE_CACHE_SIZE=25000
AST_COMPRESSION_LEVEL=6
AST_SAMPLE_PER_CATEGORY=4
AST_NICAD_BAND_EDGES="100,95,90,80,70"
# Preserve both v3 summary-clonetype.md and v4 summary-clonetype-h08-v4.md.
# v5 report is fixed-width, split into two compact tables; T2 total is a subtotal.
AST_REPORT_NAME="summary-clonetype-h08-v8.md"
# J-03d v8 dual positive selection; v5 census is immutable reference.
POSITIVE_IMPL="$ROOT/src/py3/j-03d-fixed-positives.py"
POSITIVE_SOURCE_CENSUS_NAME="summary-clonetype-h08-v5.md"
POSITIVE_SOURCE_CENSUS_SHA256="f510a87fee06c4759b7c2d1b61a4e467cf28f51fe949488dd3a9d083e15f5b1e"
POSITIVE_BALANCE_POLICY="all_available_types"
POSITIVE_BALANCE_TYPES="TYPE_1_SIGNATURE,TYPE_2_CONSISTENT_SIGNATURE,TYPE_2_BLIND_SIGNATURE,TYPE_3_STRUCTURAL_CANDIDATE"
POSITIVE_TARGET_TYPE_RATIO="native"
POSITIVE_MISSING_TYPE_POLICY="preserve_missing_types_with_warning"
POSITIVE_TYPE2_BLIND_MIN=1
POSITIVE_SAMPLE_SEED=20261118
POSITIVE_DEDUP_STRATEGY="unordered_source_sha256_pair"
POSITIVE_RECORD_SCHEMA="function_refs_v8"
POSITIVE_FUNCTION_ID_SOURCE_FIELD="fragment_id"
POSITIVE_PUBLIC_NAME="fixed_positive_clonepairs.jsonl"
POSITIVE_DETAILS_DIR="details"
POSITIVE_DETAILS_NAME="fixed_positive_clonepairs.jsonl"
POSITIVE_MANIFEST_NAME="fixed_positive_clonepairs.manifest.json"
POSITIVE_REPORT_NAME="summary-fixed-positive-clonepairs-v8.md"
POSITIVE_LAYOUT="root"
POSITIVE_PROGRESS_PAIRS=100000
POSITIVE_SQLITE_CACHE_KIB=32768
POSITIVE_CREATE_OUTPUTS="true"
# Mode-specific configuration: explicitly passed to Python through COMMON below.
case "${1:-}" in
  --positiveclone-sameratio-type|--positiveclone-sameratio-type-verify)
    POSITIVE_BALANCE_POLICY="equal_available_types"
    POSITIVE_TARGET_TYPE_RATIO="1,1,1"
    POSITIVE_MISSING_TYPE_POLICY="equal_present_types_with_warning"
    POSITIVE_LAYOUT="sameratio-type"
    POSITIVE_REPORT_NAME="summary-fixed-positive-clonepairs-sameratio-type-v8.md"
    ;;
  --positiveclone|--positiveclone-verify)
    POSITIVE_BALANCE_POLICY="all_available_types"
    POSITIVE_TARGET_TYPE_RATIO="native"
    POSITIVE_MISSING_TYPE_POLICY="preserve_missing_types_with_warning"
    POSITIVE_LAYOUT="root"
    POSITIVE_REPORT_NAME="summary-fixed-positive-clonepairs-v8.md"
    ;;
esac
# ----------------------------- Logging configuration ------------------------
LOG_DIR="$ROOT/logs/j-03d"
LOG_PREFIX="j-03d-v8-positive-clones"
LOG_TIMESTAMP_FORMAT="%Y%m%d-%H%M%S-%6N"
LOG_TIME_ZONE="America/Chicago"
LOG_TEE_APPEND="true"
LOG_CAPTURE_STDOUT="true"
LOG_CAPTURE_STDERR="true"
LOG_STEP_TIMING="true"
LOG_SHOW_TERMINAL="true"
export TZ="$LOG_TIME_ZONE"
export LC_ALL=C
now() { date '+%Y-%m-%dT%H:%M:%S%:z'; }
usage() {
  echo "Usage: bash src/sh3/j-03d-positive-clones.sh --inspect|--smoke|--subject SUBJECT|--all|--verify|--clonetypes-smoke|--clonetypes|--positiveclone|--positiveclone-verify|--positiveclone-sameratio-type|--positiveclone-sameratio-type-verify" >&2
  exit 2
}
(( $# >= 1 )) || usage
MODE="$1"; shift
SUBJECT=""
case "$MODE" in
  --inspect|--smoke|--all|--verify|--clonetypes-smoke|--clonetypes|--positiveclone|--positiveclone-verify|--positiveclone-sameratio-type|--positiveclone-sameratio-type-verify) (( $# == 0 )) || usage ;;
  --subject)
    (( $# == 1 )) || usage
    SUBJECT="$1"
    [[ "$SUBJECT" =~ ^[a-z0-9][a-z0-9_.-]*__[a-z0-9][a-z0-9_.-]*$ ]] || usage ;;
  *) usage ;;
esac
[[ -f "$PYTHON_SCRIPT" ]] || { echo "Missing implementation: $PYTHON_SCRIPT" >&2; exit 2; }
if [[ "$MODE" == "--clonetypes" || "$MODE" == "--clonetypes-smoke" || "$MODE" == "--positiveclone" || "$MODE" == "--positiveclone-verify" || "$MODE" == "--positiveclone-sameratio-type" || "$MODE" == "--positiveclone-sameratio-type-verify" ]]; then
  for file in "$AST_IMPL" "$H07_PYTHON" "$H08_PYTHON"; do
    [[ -f "$file" ]] || { echo "Missing AST implementation/reference: $file" >&2; exit 2; }
  done
fi
if [[ "$MODE" == "--positiveclone" || "$MODE" == "--positiveclone-verify" || "$MODE" == "--positiveclone-sameratio-type" || "$MODE" == "--positiveclone-sameratio-type-verify" ]]; then
  [[ -f "$POSITIVE_IMPL" ]] || { echo "Missing positive implementation: $POSITIVE_IMPL" >&2; exit 2; }
fi
command -v flock >/dev/null 2>&1 || { echo "flock command required" >&2; exit 2; }
mkdir -p "$LOG_DIR" "$RESULT_ROOT" "$OUTPUT_ROOT"
TAG="$(date "+$LOG_TIMESTAMP_FORMAT")-$$-$(od -An -tx1 -N6 /dev/urandom | tr -d ' \n')"
RUN_DIR="$RESULT_ROOT/$(date '+%Y%m%d-%H%M%S')-${MODE#--}-$TAG"
mkdir "$RUN_DIR"
LOG_FILE="$LOG_DIR/$LOG_PREFIX-$TAG.log"
( set -C; : > "$LOG_FILE" ) || { echo "Log collision: $LOG_FILE" >&2; exit 2; }
exec > >(tee -a "$LOG_FILE") 2>&1

START_EPOCH="$(date +%s)"
START_TIME="$(now)"
CURRENT_STEP="init"
CURRENT_SUBJECT="global"
FINAL_STATUS="PASS"
finish() {
  local ec="$?"
  trap - EXIT
  if (( ec != 0 )); then FINAL_STATUS="FAIL"; fi
  echo "[$(now)] $FINAL_STATUS FINAL experiment=$EXPERIMENT version=$VERSION mode=$MODE subject=$CURRENT_SUBJECT exit=$ec start=$START_TIME end=$(now) elapsed_sec=$(($(date +%s)-START_EPOCH)) run=$RUN_DIR log=$LOG_FILE input=$INPUT_ROOT output=$OUTPUT_ROOT"
  exit "$ec"
}
trap finish EXIT
trap 'exit 130' INT TERM

exec 9>"$ROOT/data/clone-detection/.j03d-positive-clones.lock"
flock -n 9 || { echo "[$(now)] FAIL another J-03d writer is active" >&2; exit 2; }
COMMON=(
  --experiment "$EXPERIMENT" --version "$VERSION"
  --project-root "$ROOT" --input-root "$INPUT_ROOT"
  --expected-input-relative "$EXPECTED_INPUT_RELATIVE"
  --source-root "$SOURCE_ROOT" --output-root "$OUTPUT_ROOT"
  --expected-output-relative "$EXPECTED_OUTPUT_RELATIVE" --run-dir "$RUN_DIR"
  --expected-subjects "$EXPECTED_SUBJECTS" --expected-total-pairs "$EXPECTED_TOTAL_PAIRS"
  --expected-input-experiment "$EXPECTED_INPUT_EXPERIMENT"
  --expected-input-version "$EXPECTED_INPUT_VERSION"
  --expected-input-phase "$EXPECTED_INPUT_PHASE"
  --expected-input-status "$EXPECTED_INPUT_STATUS"
  --expected-preproc-experiment "$EXPECTED_PREPROC_EXPERIMENT"
  --expected-preproc-version "$EXPECTED_PREPROC_VERSION"
  --expected-preproc-status "$EXPECTED_PREPROC_STATUS"
  --expected-nicad-config "$EXPECTED_NICAD_CONFIG"
  --expected-nicad-threshold "$EXPECTED_NICAD_THRESHOLD"
  --expected-detector "$EXPECTED_DETECTOR" --expected-language "$EXPECTED_LANGUAGE"
  --positive-label "$POSITIVE_LABEL"
  --code-encoding "$CODE_ENCODING" --code-errors "$CODE_ERRORS"
  --time-zone "$LOG_TIME_ZONE"
  --progress-every-fragments "$PROGRESS_EVERY_FRAGMENTS"
  --progress-every-pairs "$PROGRESS_EVERY_PAIRS"
  --accepted-output-versions "$ACCEPTED_OUTPUT_VERSIONS"
  --clonetype-t1-label "$CLONETYPE_T1_LABEL"
  --clonetype-t2-label "$CLONETYPE_T2_LABEL"
  --clonetype-t3-label "$CLONETYPE_T3_LABEL"
  --clonetype-t4-label "$CLONETYPE_T4_LABEL"
  --clonetype-min-similarity "$CLONETYPE_MIN_SIMILARITY"
  --clonetype-max-similarity "$CLONETYPE_MAX_SIMILARITY"
  --clonetype-report-name "$CLONETYPE_REPORT_NAME"
  --clonetype-progress-every-pairs "$CLONETYPE_PROGRESS_EVERY_PAIRS"
  --h07-python "$H07_PYTHON" --h08-python "$H08_PYTHON" --ast-impl "$AST_IMPL"
  --ast-parser-version "$AST_PARSER_VERSION" --ast-grammar-version "$AST_GRAMMAR_VERSION"
  --ast-gram-size "$AST_GRAM_SIZE" --ast-primary-threshold "$AST_PRIMARY_THRESHOLD"
  --ast-thresholds "$AST_THRESHOLDS" --ast-max-cells "$AST_MAX_COMPARISON_CELLS"
  --ast-progress-fragments "$AST_PROGRESS_FRAGMENTS" --ast-progress-pairs "$AST_PROGRESS_PAIRS"
  --ast-fragment-cache-size "$AST_FRAGMENT_CACHE_SIZE" --ast-score-cache-size "$AST_SCORE_CACHE_SIZE"
  --ast-compression-level "$AST_COMPRESSION_LEVEL" --ast-sample-per-category "$AST_SAMPLE_PER_CATEGORY"
  --ast-report-name "$AST_REPORT_NAME"
  --ast-nicad-band-edges "$AST_NICAD_BAND_EDGES"
  --positive-impl "$POSITIVE_IMPL"
  --positive-source-census-name "$POSITIVE_SOURCE_CENSUS_NAME"
  --positive-source-census-sha256 "$POSITIVE_SOURCE_CENSUS_SHA256"
  --positive-layout "$POSITIVE_LAYOUT"
  --positive-balance-policy "$POSITIVE_BALANCE_POLICY"
  --positive-balance-types "$POSITIVE_BALANCE_TYPES"
  --positive-target-type-ratio "$POSITIVE_TARGET_TYPE_RATIO"
  --positive-missing-type-policy "$POSITIVE_MISSING_TYPE_POLICY"
  --positive-type2-blind-min "$POSITIVE_TYPE2_BLIND_MIN"
  --positive-sample-seed "$POSITIVE_SAMPLE_SEED"
  --positive-dedup-strategy "$POSITIVE_DEDUP_STRATEGY"
  --positive-record-schema "$POSITIVE_RECORD_SCHEMA"
  --positive-function-id-source-field "$POSITIVE_FUNCTION_ID_SOURCE_FIELD"
  --positive-public-name "$POSITIVE_PUBLIC_NAME"
  --positive-details-dir "$POSITIVE_DETAILS_DIR"
  --positive-details-name "$POSITIVE_DETAILS_NAME"
  --positive-manifest-name "$POSITIVE_MANIFEST_NAME"
  --positive-report-name "$POSITIVE_REPORT_NAME"
  --positive-progress-pairs "$POSITIVE_PROGRESS_PAIRS"
  --positive-sqlite-cache-kib "$POSITIVE_SQLITE_CACHE_KIB"
  --positive-create-outputs "$POSITIVE_CREATE_OUTPUTS"
)
runpy() {
  local action="$1" subject="$2"
  "$PYTHON_BIN" "$PYTHON_SCRIPT" "${COMMON[@]}" --mode "$action" --subject "$subject"
}
step() {
  local name="$1"; shift
  local start rc=0
  CURRENT_STEP="$name"
  start="$(date +%s)"
  echo "[$(now)] START step=$name subject=$CURRENT_SUBJECT"
  "$@" || rc=$?
  if (( rc == 0 )); then
    echo "[$(now)] PASS step=$name subject=$CURRENT_SUBJECT elapsed_sec=$(($(date +%s)-start))"
  else
    echo "[$(now)] FAIL step=$name subject=$CURRENT_SUBJECT exit=$rc elapsed_sec=$(($(date +%s)-start))"
  fi
  return "$rc"
}
subjects() {
  local path
  SUBJECTS=()
  for path in "$INPUT_ROOT"/*__*; do
    [[ -d "$path" && ! -L "$path" ]] || continue
    SUBJECTS+=("${path##*/}")
  done
  (( ${#SUBJECTS[@]} == EXPECTED_SUBJECTS )) || {
    echo "AFTER_SUBJECT_COUNT_MISMATCH: ${#SUBJECTS[@]} != $EXPECTED_SUBJECTS" >&2
    return 2
  }
  mapfile -t SUBJECTS < <(printf '%s\n' "${SUBJECTS[@]}" | LC_ALL=C sort)
}

echo "[$(now)] START experiment=$EXPERIMENT version=$VERSION mode=$MODE run=$RUN_DIR log=$LOG_FILE"
echo "[$(now)] CONFIG input=$INPUT_ROOT input_phase=$EXPECTED_INPUT_PHASE source=$SOURCE_ROOT output=$OUTPUT_ROOT expected_subjects=$EXPECTED_SUBJECTS expected_pairs=$EXPECTED_TOTAL_PAIRS detector=$EXPECTED_DETECTOR config=$EXPECTED_NICAD_CONFIG threshold=$EXPECTED_NICAD_THRESHOLD preproc_version=$EXPECTED_PREPROC_VERSION label=$POSITIVE_LABEL encoding=$CODE_ENCODING errors=$CODE_ERRORS time_zone=$LOG_TIME_ZONE progress_fragments=$PROGRESS_EVERY_FRAGMENTS progress_pairs=$PROGRESS_EVERY_PAIRS accepted_output_versions=$ACCEPTED_OUTPUT_VERSIONS ast_primary=$AST_PRIMARY_THRESHOLD ast_gram_size=$AST_GRAM_SIZE ast_max_cells=$AST_MAX_COMPARISON_CELLS ast_report=$AST_REPORT_NAME ast_nicad_bands=$AST_NICAD_BAND_EDGES clonetype_min=$CLONETYPE_MIN_SIMILARITY clonetype_max=$CLONETYPE_MAX_SIMILARITY clonetype_report=$CLONETYPE_REPORT_NAME clonetype_progress=$CLONETYPE_PROGRESS_EVERY_PAIRS positive_balance=$POSITIVE_BALANCE_POLICY positive_layout=$POSITIVE_LAYOUT positive_seed=$POSITIVE_SAMPLE_SEED positive_record_schema=$POSITIVE_RECORD_SCHEMA function_id_source=$POSITIVE_FUNCTION_ID_SOURCE_FIELD positive_report=$POSITIVE_REPORT_NAME"
case "$MODE" in
  --inspect)
    CURRENT_SUBJECT="global"
    step preflight runpy inspect "$SMOKE_SUBJECT" ;;
  --smoke)
    CURRENT_SUBJECT="$SMOKE_SUBJECT"
    step smoke_export runpy export "$SMOKE_SUBJECT"
    step smoke_verify runpy verify "$SMOKE_SUBJECT" ;;
  --subject)
    CURRENT_SUBJECT="$SUBJECT"
    step subject_export runpy export "$SUBJECT"
    step subject_verify runpy verify "$SUBJECT" ;;
  --all)
    CURRENT_SUBJECT="global"
    step preflight runpy inspect "$SMOKE_SUBJECT"
    subjects
    idx=0
    for SUBJECT in "${SUBJECTS[@]}"; do
      idx=$((idx+1)); CURRENT_SUBJECT="$SUBJECT"
      echo "[$(now)] SUBJECT_START index=$idx/$EXPECTED_SUBJECTS subject=$SUBJECT"
      step subject_export runpy export "$SUBJECT"
      echo "[$(now)] SUBJECT_DONE index=$idx/$EXPECTED_SUBJECTS subject=$SUBJECT"
    done
    CURRENT_SUBJECT="global"
    step aggregate runpy summarize "$SMOKE_SUBJECT" ;;
  --clonetypes-smoke)
    CURRENT_SUBJECT="$SMOKE_SUBJECT"
    step ast_clonetype_smoke runpy clonetypes-smoke "$SMOKE_SUBJECT" ;;
  --clonetypes)
    CURRENT_SUBJECT="global"
    step ast_clonetype_census runpy clonetypes "$SMOKE_SUBJECT" ;;
  --positiveclone)
    CURRENT_SUBJECT="global"
    step all_positive_prepare runpy positiveclone "$SMOKE_SUBJECT" ;;
  --positiveclone-sameratio-type)
    CURRENT_SUBJECT="global"
    step sameratio_positive_prepare runpy positiveclone-sameratio-type "$SMOKE_SUBJECT" ;;
  --positiveclone-verify)
    CURRENT_SUBJECT="global"
    step all_positive_verify runpy positiveclone-verify "$SMOKE_SUBJECT" ;;
  --positiveclone-sameratio-type-verify)
    CURRENT_SUBJECT="global"
    step sameratio_positive_verify runpy positiveclone-sameratio-type-verify "$SMOKE_SUBJECT" ;;
  --verify)
    subjects
    idx=0
    for SUBJECT in "${SUBJECTS[@]}"; do
      idx=$((idx+1)); CURRENT_SUBJECT="$SUBJECT"
      echo "[$(now)] VERIFY_START index=$idx/$EXPECTED_SUBJECTS subject=$SUBJECT"
      step subject_verify runpy verify "$SUBJECT"
    done
    CURRENT_SUBJECT="global"
    step aggregate runpy summarize "$SMOKE_SUBJECT" ;;
esac
