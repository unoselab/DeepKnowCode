#!/usr/bin/env bash
# DeepKnowCode J-03e v1 - independent shell configuration, execution and Chicago logging.
# Run from repository root, or by absolute path from elsewhere.
set -Eeuo pipefail

EXPERIMENT="j-03e"
VERSION="v1"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
PYTHON_BIN="python3"
PYTHON_SCRIPT="$ROOT/src/py3/j-03e-negative-clones.py"
POSITIVE_ROOT="$ROOT/data/clone-detection/3pos-clones"
NEGATIVE_ROOT="$ROOT/data/clone-detection/4neg-clones"
RESULT_ROOT="$ROOT/results/j-03e"
LOG_DIR="$ROOT/logs/j-03e"
LOG_PREFIX="j-03e-v1-negative-clones"
LOG_TIMESTAMP_FORMAT="%Y%m%d-%H%M%S-%N"
LOG_TEE_APPEND="true"
TIME_ZONE="America/Chicago"

# Immutable input provenance, verified against actual J-03d v8 output report.
REFERENCE_SUMMARY_NAME="summary-fixed-positive-clonepairs-v8.md"
REFERENCE_SUMMARY_SHA256="137c02c88300eaebb1240a727c0025789f8a43c56ec101b9b738df9355961a0d"
SOURCE_MANIFEST_NAME="manifest.json"
FRAGMENT_NAME="fragments.jsonl"
ORIGINAL_POSITIVE_CSV_NAME="positive_pairs.csv"
FIXED_POSITIVE_PUBLIC_NAME="fixed_positive_clonepairs.jsonl"
POSITIVE_DETAILS_DIR="details"
FIXED_POSITIVE_DETAILS_NAME="fixed_positive_clonepairs.jsonl"
FIXED_POSITIVE_MANIFEST_NAME="fixed_positive_clonepairs.manifest.json"
EXPECTED_SUBJECTS=47
EXPECTED_TOTAL_POSITIVES=2686210
POSITIVE_SELECTION_POLICY="all_available_types"
POSITIVE_RECORD_SCHEMA="function_refs_v8"
POSITIVE_LABEL=1
NEGATIVE_LABEL=0

# Exact H-07/H-08 implementation and versions from J-03d v8.
H07_PYTHON="$ROOT/src/py2/h-07-audit-java-tree-sitter.py"
H08_PYTHON="$ROOT/src/py2/h-08-analyze-java-clone-types.py"
PARSER_VERSION="0.25.2"
GRAMMAR_VERSION="0.23.5"
H07_EXPECTED_SHA256="9b37ac5c17940ba260696290f197e9f3298892ace1afaca5f7c1953b7fd029de"
H08_EXPECTED_SHA256="5213c1077ec59935316580a6c3a3b99556ae6be497f99de9bb40189c01f94df4"
AST_GRAM_SIZE=5
AST_MAX_CELLS=4000000
AST_SCORE_CACHE_SIZE=25000
POSITIVE_AST_THRESHOLD="0.82"
NEGATIVE_SIMILARITY_MAX="0.20"      # strict: score < 0.20
CODE_ENCODING="utf-8"
CODE_ERRORS="surrogateescape"

# Deterministic attempt-budgeted unique Function-ID-pair permutation.
SAMPLE_SEED=20261119
ATTEMPTS_PER_TARGET=40
ATTEMPT_LIMIT_CAP=100000000
FEASIBILITY_PROBE_PAIRS=2000
PROGRESS_EVERY=100000
SQLITE_CACHE_KIB=32768

PUBLIC_NAME="fixed_negative_clonepairs.jsonl"
DETAIL_DIR="details"
DETAIL_NAME="fixed_negative_clonepairs.jsonl"
MANIFEST_NAME="fixed_negative_clonepairs.manifest.json"
REPORT_NAME="summary-fixed-negative-clonepairs-v1.md"

usage() {
  cat <<'USAGE'
Usage:
  bash src/sh3/j-03e-negative-clones.sh --negativeclone-feasibility [--subject SUBJECT]
  bash src/sh3/j-03e-negative-clones.sh --negativeclone [--subject SUBJECT]
  bash src/sh3/j-03e-negative-clones.sh --negativeclone-verify [--subject SUBJECT]
J-03e v1 never changes the 3pos-clones inputs. Outputs require AST score < 0.20.
USAGE
}

if (( $# < 1 )); then usage >&2; exit 2; fi
case "$1" in
  --negativeclone-feasibility) MODE="negativeclone-feasibility" ;;
  --negativeclone) MODE="negativeclone" ;;
  --negativeclone-verify) MODE="negativeclone-verify" ;;
  -h|--help) usage; exit 0 ;;
  *) usage >&2; printf 'INVALID_MODE %s\n' "$1" >&2; exit 2 ;;
esac
shift
SUBJECT="ALL"
if (( $# > 0 )); then
  if [[ "$1" == "--subject" && $# -eq 2 ]]; then
    SUBJECT="$2"
    [[ "$SUBJECT" =~ ^[a-z0-9][a-z0-9_.-]*(__[a-z0-9_.-]+)?$ ]] || {
      printf 'INVALID_SUBJECT %s\n' "$SUBJECT" >&2; exit 2;
    }
  else
    usage >&2; exit 2
  fi
fi

export TZ="$TIME_ZONE"
mkdir -p "$LOG_DIR" "$RESULT_ROOT" "$NEGATIVE_ROOT"
STAMP="$(date +"$LOG_TIMESTAMP_FORMAT")"
RUN_TAG="${STAMP}-pid$BASHPID-r$RANDOM"
RUN_DIR="$RESULT_ROOT/$RUN_TAG-$MODE"
LOG_FILE="$LOG_DIR/$LOG_PREFIX-$RUN_TAG-$MODE.log"
mkdir "$RUN_DIR"
# noclobber prevents accidental reuse of any old log file.
( set -C; : > "$LOG_FILE" ) || { echo "LOG_CREATE_FAILED $LOG_FILE" >&2; exit 1; }

chicago() { date '+%Y-%m-%dT%H:%M:%S%:z'; }
log() { printf '[%s] %s\n' "$(chicago)" "$*" | tee -a "$LOG_FILE"; }
START_EPOCH="$(date +%s)"
FINAL_STATUS="FAIL"
FINAL_EXIT=1
on_exit() {
  local status="$?" end
  trap - EXIT
  end="$(date +%s)"
  if (( status == 0 )); then FINAL_STATUS="PASS"; else FINAL_STATUS="FAIL"; fi
  FINAL_EXIT="$status"
  log "$FINAL_STATUS FINAL experiment=$EXPERIMENT version=$VERSION mode=$MODE subject=$SUBJECT exit=$FINAL_EXIT start_epoch=$START_EPOCH end=$(chicago) elapsed_sec=$((end-START_EPOCH)) run=$RUN_DIR log=$LOG_FILE input=$POSITIVE_ROOT output=$NEGATIVE_ROOT"
  exit "$status"
}
trap on_exit EXIT

log "START experiment=$EXPERIMENT version=$VERSION mode=$MODE subject=$SUBJECT run=$RUN_DIR log=$LOG_FILE"
log "CONFIG input=$POSITIVE_ROOT output=$NEGATIVE_ROOT expected_subjects=$EXPECTED_SUBJECTS expected_npos=$EXPECTED_TOTAL_POSITIVES ref_summary_sha256=$REFERENCE_SUMMARY_SHA256 h07=$H07_PYTHON h08=$H08_PYTHON parser=$PARSER_VERSION grammar=$GRAMMAR_VERSION ast_gram=$AST_GRAM_SIZE ast_max_cells=$AST_MAX_CELLS positive_ast_threshold=$POSITIVE_AST_THRESHOLD negative_similarity_exclusive=$NEGATIVE_SIMILARITY_MAX seed=$SAMPLE_SEED attempts_per_target=$ATTEMPTS_PER_TARGET attempt_cap=$ATTEMPT_LIMIT_CAP feasibility_probe=$FEASIBILITY_PROBE_PAIRS progress=$PROGRESS_EVERY time_zone=$TIME_ZONE"
log "START step=preflight"
[[ -f "$PYTHON_SCRIPT" && -f "$H07_PYTHON" && -f "$H08_PYTHON" ]] || {
  log "FAIL step=preflight reason=required_python_or_ast_reference_missing"; exit 1;
}
[[ -d "$POSITIVE_ROOT" ]] || { log "FAIL step=preflight reason=positive_root_missing"; exit 1; }
log "PASS step=preflight"

log "START step=$MODE"
set +e
"$PYTHON_BIN" -u "$PYTHON_SCRIPT" \
  --experiment "$EXPERIMENT" --version "$VERSION" --mode "$MODE" --subject "$SUBJECT" \
  --time-zone "$TIME_ZONE" --positive-root "$POSITIVE_ROOT" \
  --negative-root "$NEGATIVE_ROOT" --run-dir "$RUN_DIR" \
  --reference-summary-name "$REFERENCE_SUMMARY_NAME" \
  --reference-summary-sha256 "$REFERENCE_SUMMARY_SHA256" \
  --source-manifest-name "$SOURCE_MANIFEST_NAME" \
  --fragment-name "$FRAGMENT_NAME" \
  --original-positive-csv-name "$ORIGINAL_POSITIVE_CSV_NAME" \
  --fixed-positive-public-name "$FIXED_POSITIVE_PUBLIC_NAME" \
  --positive-details-dir "$POSITIVE_DETAILS_DIR" \
  --fixed-positive-details-name "$FIXED_POSITIVE_DETAILS_NAME" \
  --fixed-positive-manifest-name "$FIXED_POSITIVE_MANIFEST_NAME" \
  --expected-subjects "$EXPECTED_SUBJECTS" \
  --expected-total-positives "$EXPECTED_TOTAL_POSITIVES" \
  --positive-policy "$POSITIVE_SELECTION_POLICY" \
  --positive-schema "$POSITIVE_RECORD_SCHEMA" \
  --positive-label "$POSITIVE_LABEL" --negative-label "$NEGATIVE_LABEL" \
  --h07-python "$H07_PYTHON" --h08-python "$H08_PYTHON" \
  --parser-version "$PARSER_VERSION" --grammar-version "$GRAMMAR_VERSION" \
  --h07-expected-sha256 "$H07_EXPECTED_SHA256" --h08-expected-sha256 "$H08_EXPECTED_SHA256" \
  --positive-ast-threshold "$POSITIVE_AST_THRESHOLD" \
  --ast-gram-size "$AST_GRAM_SIZE" --ast-max-cells "$AST_MAX_CELLS" \
  --ast-score-cache-size "$AST_SCORE_CACHE_SIZE" \
  --negative-similarity-max "$NEGATIVE_SIMILARITY_MAX" \
  --code-encoding "$CODE_ENCODING" --code-errors "$CODE_ERRORS" \
  --sample-seed "$SAMPLE_SEED" \
  --attempts-per-target "$ATTEMPTS_PER_TARGET" --attempt-limit-cap "$ATTEMPT_LIMIT_CAP" \
  --feasibility-probe-pairs "$FEASIBILITY_PROBE_PAIRS" \
  --progress-every "$PROGRESS_EVERY" --sqlite-cache-kib "$SQLITE_CACHE_KIB" \
  --public-name "$PUBLIC_NAME" --detail-dir "$DETAIL_DIR" \
  --detail-name "$DETAIL_NAME" --manifest-name "$MANIFEST_NAME" \
  --report-name "$REPORT_NAME" \
  2>&1 | tee -a "$LOG_FILE"
PIPE_RESULTS=("${PIPESTATUS[@]}")
PY_STATUS=${PIPE_RESULTS[0]}
TEE_STATUS=${PIPE_RESULTS[1]}
set -e
if (( PY_STATUS != 0 || TEE_STATUS != 0 )); then
  log "FAIL step=$MODE python_exit=$PY_STATUS tee_exit=$TEE_STATUS run=$RUN_DIR"
  exit 1
fi
log "PASS step=$MODE run=$RUN_DIR"
