#!/usr/bin/env bash
set -Eeuo pipefail
export TZ='America/Chicago'
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
EXPERIMENT='j-01d'
SCRIPT_VERSION='v3'
LOG_DIR="$ROOT/logs/$EXPERIMENT"
RESULT_DIR="$ROOT/results/$EXPERIMENT"
LOG_PREFIX="$EXPERIMENT-$SCRIPT_VERSION"
LOG_TIMESTAMP_FORMAT='+%Y-%m%d-%H%M%S'
LOG_TEE='true'
LOG_STDERR='true'
GROUND_TRUTH_ROOT="$ROOT/data/evaluation/ground_truth/python"
SOURCE_FILE="$GROUND_TRUTH_ROOT/verified_pairs.jsonl"
ORIGINAL_DETAILS_DIR='details'
REFACTORABILITY_DIR='refactorability'
CLONE_DETAILS_DIR='clone_details'
DATASET_VERSIONS_JSON='["v1","v2"]'
PROJECT_OWNERS_JSON='{"Python":"TheAlgorithms","ansible":"ansible","boto3":"boto","core":"home-assistant","django":"django","gensim":"piskvorky","h2ogpt":"h2oai","horizon":"openstack","ironic":"openstack","keras":"keras-team","keystone":"openstack","libcloud":"apache","matplotlib":"matplotlib","neutron":"openstack","nova":"openstack","pandas":"pandas-dev","paramiko":"paramiko","salt":"saltstack","scikit-learn":"scikit-learn","sunnypilot":"sunnypilot","swift":"openstack","sympy":"sympy","web2py":"web2py","werkzeug":"pallets"}'
EXPECTED_COUNTS_JSON='{"v1":7942,"v2":7320}'
DETAIL_COLUMNS_JSON='["pair_left_func","pair_right_func","pair_id","project","gold_label","label_source","source_metadata_json"]'
REPORT_FILENAME='j-01d-clone-details-report.json'
usage() { echo "Usage: bash src/sh3/j-01d-organize-python-clone-details.sh --inspect|--build" >&2; exit 2; }
[[ "$#" -eq 1 ]] || usage
MODE="$1"
case "$MODE" in --inspect|--build) ;; *) usage ;; esac
mkdir -p "$LOG_DIR" "$RESULT_DIR"
START_EPOCH="$(date +%s)"
START_CHICAGO="$(date -Iseconds)"
STAMP="$(date "$LOG_TIMESTAMP_FORMAT")"
RUN_DIR="$(mktemp -d "$RESULT_DIR/${STAMP}-XXXXXX")"
LOG_FILE="$LOG_DIR/$LOG_PREFIX-$STAMP-${BASHPID}-${RUN_DIR##*-}.log"
( set -o noclobber; : > "$LOG_FILE" )
exec > >(tee -a "$LOG_FILE") 2>&1
step_name='initialization';step_start="$START_EPOCH";finished=0
log() { printf '[%s] %s\n' "$(date -Iseconds)" "$*"; }
start_step() { step_name="$1";step_start="$(date +%s)";log "START step=$step_name"; }
pass_step() { log "PASS step=$step_name elapsed_sec=$(($(date +%s)-step_start))"; }
finish() {
 local ec=$?
 if [[ $finished -eq 1 ]]; then return; fi
 finished=1
 local status='PASS'
 if [[ "$ec" -ne 0 ]]; then status='FAIL';log "FAIL step=$step_name elapsed_sec=$(($(date +%s)-step_start))"; fi
 log "$status FINAL experiment=$EXPERIMENT version=$SCRIPT_VERSION mode=$MODE elapsed_sec=$(($(date +%s)-START_EPOCH)) exit=$ec"
 printf "LOG='%s' RUN='%s' START_CHICAGO='%s' END_CHICAGO='%s'\n" "$LOG_FILE" "$RUN_DIR" "$START_CHICAGO" "$(date -Iseconds)"
}
trap finish EXIT
log "START experiment=$EXPERIMENT version=$SCRIPT_VERSION mode=$MODE log=$LOG_FILE run=$RUN_DIR"
start_step validate_files
[[ -f "$ROOT/src/py3/j-01d-organize-python-clone-details.py" ]] || { log 'FAIL missing Python script';exit 1; }
[[ -f "$SOURCE_FILE" ]] || { log "FAIL missing source=$SOURCE_FILE";exit 1; }
pass_step
start_step "$MODE"
"$PYTHON_BIN" "$ROOT/src/py3/j-01d-organize-python-clone-details.py" \
 --root "$GROUND_TRUTH_ROOT" --source "$SOURCE_FILE" --run-dir "$RUN_DIR" \
 --mode "${MODE#--}" --experiment "$EXPERIMENT" --version "$SCRIPT_VERSION" \
 --report-name "$REPORT_FILENAME" --old-dir "$ORIGINAL_DETAILS_DIR" \
 --renamed-dir "$REFACTORABILITY_DIR" --details-dir "$CLONE_DETAILS_DIR" \
 --versions-json "$DATASET_VERSIONS_JSON" --owner-map-json "$PROJECT_OWNERS_JSON" \
 --expected-pairs-json "$EXPECTED_COUNTS_JSON" --expected-detail-cols-json "$DETAIL_COLUMNS_JSON"
pass_step
