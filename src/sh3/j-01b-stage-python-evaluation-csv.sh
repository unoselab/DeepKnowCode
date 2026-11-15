#!/usr/bin/env bash
# J-01b v12: owner_project Ground Truth CSV migration, standalone shell config.
set -Eeuo pipefail
export TZ='America/Chicago'
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"

# Every effective experiment setting is declared here and passed to Python.
EXPERIMENT='j-01b'
VERSION='v12'
PYTHON_BIN='python'
ENGINE="$ROOT/src/py3/j-01b-stage-python-evaluation-csv.py"
SOURCE_PAIRS="$ROOT/data/evaluation/ground_truth/python/verified_pairs.jsonl"
CODE_JSONL="$ROOT/data/evaluation/code/python.jsonl"
OUTPUT_DIR="$ROOT/data/evaluation/ground_truth/python"
EXPECTED_PAIRS='7942'
EXPECTED_FUNCTIONS='7163'
EXPECTED_PROJECTS='24'
EXPECTED_POSITIVE='3971'
EXPECTED_NEGATIVE='3971'
EXPECTED_SOURCE_SHA256='2d171cdee52f6ab8229853b0cc5590b71e1a7dd6919a417dd88fb17d0ec0cb8c'
EXPECTED_CODE_SHA256='8b991eff9a6cb55bb42bc29e10a9449b8e188ddf1ed61dea725352cf3eff5b96'
CODE_ID_FIELD='id'
CODE_VALUE_FIELD='code'
SOURCE_CODE_FIELD='source_code'
CODE_ID_PREFIX='py-sha256:'
PAIR_ID_FIELD='pair_id'
PROJECT_FIELD='project'
LABEL_FIELD='gold_label'
ELIGIBLE_FIELD='evaluation_eligible'
LEFT_FIELD='left'
RIGHT_FIELD='right'
OUTPUT_LEFT_FIELD='pair_left_func'
OUTPUT_RIGHT_FIELD='pair_right_func'
OUTPUT_LABEL_FIELD='clone_label'
OUTPUT_COLUMNS_JSON='["pair_left_func","pair_right_func","clone_label"]'
CSV_NAME_PATTERN='{owner}_{project}.csv'

# Verified from git remote origin for each of 24 repositories, documented by
# the project's confirmed Dataset README. Filename components are normalized to lowercase in Python.
OWNER_MAP_SOURCE_URL='https://github.com/unoselab/ai-code-complexity-study-local/blob/main/README-1111-DeepKnowCode-Dataset.md'
PROJECT_OWNERS_JSON='{"Python":"TheAlgorithms","ansible":"ansible","boto3":"boto","core":"home-assistant","django":"django","gensim":"piskvorky","h2ogpt":"h2oai","horizon":"openstack","ironic":"openstack","keras":"keras-team","keystone":"openstack","libcloud":"apache","matplotlib":"matplotlib","neutron":"openstack","nova":"openstack","pandas":"pandas-dev","paramiko":"paramiko","salt":"saltstack","scikit-learn":"scikit-learn","sunnypilot":"sunnypilot","swift":"openstack","sympy":"sympy","web2py":"web2py","werkzeug":"pallets"}'

# Legacy v10 files are archived byte-for-byte before publishing new CSVs.
LEGACY_CSV_NAME_PATTERN='pred_{project}_with_refactorability.csv'
LEGACY_DETAIL_NAME_PATTERN='pred_{project}_with_refactorability_detail.csv'
LEGACY_COLUMNS_JSON='["pair_left_func","pair_right_func","clone_predict","pair_left_in","pair_right_in","pair_left_out","pair_right_out","pair_inType_ok","pair_outType_ok","pair_type_ok","pair_left_extractable","pair_right_extractable","pair_refactorable"]'
LEGACY_DETAIL_COLUMNS_JSON='["pair_left_func","pair_right_func","pair_id","project","gold_label","label_source","source_metadata_json"]'
DETAILS_SUBDIR='details'
ARCHIVE_SUBDIR='archive'
ARCHIVE_PREFIX='j-01b-v10-' 
NULL_LITERAL='NULL'
REPORT_FILENAME='ground_truth_csv_report.json'

LOG_DIR="$ROOT/logs/j-01b"
RESULT_DIR="$ROOT/results/j-01b"
LOG_PREFIX='j-01b-v12'
LOG_TIMESTAMP_FORMAT='+%Y-%m%d-%H%M%S'
LOG_TEE='true'
LOG_STDERR='true'

MODE="${1:-}"
if [[ "$#" -ne 1 ]]; then
  echo "Usage: bash src/sh3/j-01b-stage-python-evaluation-csv.sh --inspect|--convert" >&2
  exit 2
fi
case "$MODE" in
  --inspect|--convert) ;;
  *) echo "Usage: bash src/sh3/j-01b-stage-python-evaluation-csv.sh --inspect|--convert" >&2; exit 2 ;;
esac
if [[ "$LOG_TEE" != 'true' || "$LOG_STDERR" != 'true' ]]; then
  echo 'Logging config requires tee of both stdout and stderr' >&2
  exit 2
fi
mkdir -p -- "$LOG_DIR" "$RESULT_DIR"
START_EPOCH="$(date +%s)"
START_CHICAGO="$(date -Iseconds)"
STAMP="$(date "$LOG_TIMESTAMP_FORMAT")"
RUN_DIR="$(mktemp -d "$RESULT_DIR/${STAMP}-XXXXXX")"
LOG_FILE="$LOG_DIR/${LOG_PREFIX}-${STAMP}-${BASHPID}-${RUN_DIR##*-}.log"
( set -o noclobber; : > "$LOG_FILE" )
exec > >(tee -a "$LOG_FILE") 2>&1
step_name='init'
step_start="$START_EPOCH"
start_step() {
  step_name="$1"
  step_start="$(date +%s)"
  printf '[%s] START step=%s\n' "$(date -Iseconds)" "$step_name"
}
pass_step() {
  printf '[%s] PASS step=%s elapsed_sec=%s\n' "$(date -Iseconds)" "$step_name" "$(( $(date +%s) - step_start ))"
}
finish() {
  local rc=$?
  trap - EXIT ERR
  if (( rc == 0 )); then
    printf '[%s] PASS FINAL experiment=%s version=%s mode=%s elapsed_sec=%s exit=%s\n' \
      "$(date -Iseconds)" "$EXPERIMENT" "$VERSION" "$MODE" "$(( $(date +%s) - START_EPOCH ))" "$rc"
  else
    printf '[%s] FAIL step=%s elapsed_sec=%s\n' "$(date -Iseconds)" "$step_name" "$(( $(date +%s) - step_start ))"
    printf '[%s] FAIL FINAL experiment=%s version=%s mode=%s elapsed_sec=%s exit=%s\n' \
      "$(date -Iseconds)" "$EXPERIMENT" "$VERSION" "$MODE" "$(( $(date +%s) - START_EPOCH ))" "$rc"
  fi
  printf "LOG='%s' RUN='%s' OUTPUT_DIR='%s'\n" "$LOG_FILE" "$RUN_DIR" "$OUTPUT_DIR"
  printf "START_CHICAGO='%s' END_CHICAGO='%s'\n" "$START_CHICAGO" "$(date -Iseconds)"
  exit "$rc"
}
trap finish EXIT
printf '[%s] START experiment=%s version=%s mode=%s log=%s run=%s\n' \
  "$(date -Iseconds)" "$EXPERIMENT" "$VERSION" "$MODE" "$LOG_FILE" "$RUN_DIR"

start_step validate_files
for file in "$ENGINE" "$SOURCE_PAIRS" "$CODE_JSONL"; do
  [[ -f "$file" ]] || { printf 'MISSING_FILE=%s\n' "$file" >&2; exit 2; }
done
pass_step
start_step "$MODE"
"$PYTHON_BIN" "$ENGINE" \
  --mode "${MODE#--}" \
  --source-pairs "$SOURCE_PAIRS" --code-jsonl "$CODE_JSONL" \
  --output-dir "$OUTPUT_DIR" --run-dir "$RUN_DIR" \
  --expected-pairs "$EXPECTED_PAIRS" --expected-functions "$EXPECTED_FUNCTIONS" \
  --expected-projects "$EXPECTED_PROJECTS" --expected-positive "$EXPECTED_POSITIVE" \
  --expected-negative "$EXPECTED_NEGATIVE" --expected-source-sha256 "$EXPECTED_SOURCE_SHA256" \
  --expected-code-sha256 "$EXPECTED_CODE_SHA256" \
  --code-id-field "$CODE_ID_FIELD" --code-value-field "$CODE_VALUE_FIELD" \
  --source-code-field "$SOURCE_CODE_FIELD" --code-id-prefix "$CODE_ID_PREFIX" \
  --pair-id-field "$PAIR_ID_FIELD" --project-field "$PROJECT_FIELD" \
  --label-field "$LABEL_FIELD" --eligible-field "$ELIGIBLE_FIELD" \
  --left-field "$LEFT_FIELD" --right-field "$RIGHT_FIELD" \
  --output-left-field "$OUTPUT_LEFT_FIELD" --output-right-field "$OUTPUT_RIGHT_FIELD" \
  --output-label-field "$OUTPUT_LABEL_FIELD" --output-columns-json "$OUTPUT_COLUMNS_JSON" \
  --project-owners-json "$PROJECT_OWNERS_JSON" --owner-map-source-url "$OWNER_MAP_SOURCE_URL" \
  --csv-name-pattern "$CSV_NAME_PATTERN" \
  --legacy-csv-name-pattern "$LEGACY_CSV_NAME_PATTERN" \
  --legacy-detail-name-pattern "$LEGACY_DETAIL_NAME_PATTERN" \
  --legacy-columns-json "$LEGACY_COLUMNS_JSON" \
  --legacy-detail-columns-json "$LEGACY_DETAIL_COLUMNS_JSON" \
  --details-subdir "$DETAILS_SUBDIR" --archive-subdir "$ARCHIVE_SUBDIR" \
  --archive-prefix "$ARCHIVE_PREFIX" --null-literal "$NULL_LITERAL" \
  --report-filename "$REPORT_FILENAME" \
  --experiment "$EXPERIMENT" --version "$VERSION"
pass_step
