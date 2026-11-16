#!/usr/bin/env bash
# J-01c v16: standalone preparation of immutable per-project 50/50 Python Ground Truth V2.
set -Eeuo pipefail
export TZ='America/Chicago'
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
EXPERIMENT='j-01c'
VERSION='v16'
DATASET_VERSION='V2'
PYTHON_BIN='python'
ENGINE="$ROOT/src/py3/j-01c-balance-python-ground-truth.py"
SOURCE_DIR="$ROOT/data/evaluation/ground_truth/python/v1"
OUTPUT_DIR="$ROOT/data/evaluation/ground_truth/python/v2"
EXPECTED_PROJECTS='24'
EXPECTED_PAIRS='7942'
EXPECTED_POSITIVE='3971'
EXPECTED_NEGATIVE='3971'
EXPECTED_V2_PAIRS='7320'
EXPECTED_V2_PER_CLASS='3660'
SEED='20261115'
SELECTION_STRATEGY='seeded-random'
OUTPUT_COLUMNS_JSON='["pair_left_func","pair_right_func","clone_label"]'
LEGACY_FILENAMES_JSON='["ansible_ansible.csv","apache_libcloud.csv","boto_boto3.csv","django_django.csv","h2oai_h2ogpt.csv","home-assistant_core.csv","keras-team_keras.csv","matplotlib_matplotlib.csv","openstack_horizon.csv","openstack_ironic.csv","openstack_keystone.csv","openstack_neutron.csv","openstack_nova.csv","openstack_swift.csv","pallets_werkzeug.csv","pandas-dev_pandas.csv","paramiko_paramiko.csv","piskvorky_gensim.csv","saltstack_salt.csv","scikit-learn_scikit-learn.csv","sunnypilot_sunnypilot.csv","sympy_sympy.csv","thealgorithms_python.csv","web2py_web2py.csv"]'
EXPECTED_FILENAMES_JSON='["ansible__ansible.csv","apache__libcloud.csv","boto__boto3.csv","django__django.csv","h2oai__h2ogpt.csv","home-assistant__core.csv","keras-team__keras.csv","matplotlib__matplotlib.csv","openstack__horizon.csv","openstack__ironic.csv","openstack__keystone.csv","openstack__neutron.csv","openstack__nova.csv","openstack__swift.csv","pallets__werkzeug.csv","pandas-dev__pandas.csv","paramiko__paramiko.csv","piskvorky__gensim.csv","saltstack__salt.csv","scikit-learn__scikit-learn.csv","sunnypilot__sunnypilot.csv","sympy__sympy.csv","thealgorithms__python.csv","web2py__web2py.csv"]'
OWNER_MAP_SOURCE='https://github.com/unoselab/ai-code-complexity-study-local/blob/main/README-1111-DeepKnowCode-Dataset.md'
MANIFEST_FILENAME='manifest.json'
REPORT_FILENAME='balanced_ground_truth_report.json'
LOG_DIR="$ROOT/logs/j-01c"
RESULT_DIR="$ROOT/results/j-01c"
LOG_PREFIX='j-01c-v16'
LOG_TIMESTAMP_FORMAT='+%Y-%m%d-%H%M%S'
LOG_TEE='true'
LOG_STDERR='true'
MODE="${1:-}"
if [[ "$#" -ne 1 ]]; then
  echo 'Usage: bash src/sh3/j-01c-balance-python-ground-truth.sh --inspect|--normalize-v1|--build' >&2; exit 2
fi
case "$MODE" in --inspect|--normalize-v1|--build) ;; *) echo 'Usage: bash src/sh3/j-01c-balance-python-ground-truth.sh --inspect|--normalize-v1|--build' >&2; exit 2 ;; esac
[[ "$LOG_TEE" = 'true' && "$LOG_STDERR" = 'true' ]] || { echo 'Invalid logging config' >&2; exit 2; }
mkdir -p "$LOG_DIR" "$RESULT_DIR"
START_EPOCH="$(date +%s)"
START_CHICAGO="$(date -Iseconds)"
STAMP="$(date "$LOG_TIMESTAMP_FORMAT")"
RUN_DIR="$(mktemp -d "$RESULT_DIR/${STAMP}-XXXXXX")"
LOG_FILE="$LOG_DIR/${LOG_PREFIX}-${STAMP}-${BASHPID}-${RUN_DIR##*-}.log"
( set -o noclobber; : > "$LOG_FILE" )
exec > >(tee -a "$LOG_FILE") 2>&1
step_name='init'
step_start="$START_EPOCH"
start_step() { step_name="$1"; step_start="$(date +%s)"; printf '[%s] START step=%s\n' "$(date -Iseconds)" "$step_name"; }
pass_step() { printf '[%s] PASS step=%s elapsed_sec=%s\n' "$(date -Iseconds)" "$step_name" "$(( $(date +%s) - step_start ))"; }
finish() {
  local rc=$?; trap - EXIT ERR
  if (( rc == 0 )); then
    printf '[%s] PASS FINAL experiment=%s version=%s dataset_version=%s mode=%s elapsed_sec=%s exit=%s\n' "$(date -Iseconds)" "$EXPERIMENT" "$VERSION" "$DATASET_VERSION" "$MODE" "$(( $(date +%s) - START_EPOCH ))" "$rc"
  else
    printf '[%s] FAIL step=%s elapsed_sec=%s\n' "$(date -Iseconds)" "$step_name" "$(( $(date +%s) - step_start ))"
    printf '[%s] FAIL FINAL experiment=%s version=%s dataset_version=%s mode=%s elapsed_sec=%s exit=%s\n' "$(date -Iseconds)" "$EXPERIMENT" "$VERSION" "$DATASET_VERSION" "$MODE" "$(( $(date +%s) - START_EPOCH ))" "$rc"
  fi
  printf "LOG='%s' RUN='%s' OUTPUT='%s'\n" "$LOG_FILE" "$RUN_DIR" "$OUTPUT_DIR"
  printf "START_CHICAGO='%s' END_CHICAGO='%s'\n" "$START_CHICAGO" "$(date -Iseconds)"
  exit "$rc"
}
trap finish EXIT
printf '[%s] START experiment=%s version=%s dataset_version=%s mode=%s log=%s run=%s\n' "$(date -Iseconds)" "$EXPERIMENT" "$VERSION" "$DATASET_VERSION" "$MODE" "$LOG_FILE" "$RUN_DIR"
start_step validate_files
[[ -f "$ENGINE" && -d "$SOURCE_DIR" ]] || { echo 'Missing engine or V1 source directory' >&2; exit 2; }
pass_step
start_step "$MODE"
"$PYTHON_BIN" "$ENGINE" \
  --mode "${MODE#--}" --source-dir "$SOURCE_DIR" --output-dir "$OUTPUT_DIR" --run-dir "$RUN_DIR" \
  --experiment "$EXPERIMENT" --version "$VERSION" --dataset-version "$DATASET_VERSION" \
  --expected-projects "$EXPECTED_PROJECTS" --expected-pairs "$EXPECTED_PAIRS" \
  --expected-positive "$EXPECTED_POSITIVE" --expected-negative "$EXPECTED_NEGATIVE" \
  --expected-v2-pairs "$EXPECTED_V2_PAIRS" --expected-v2-per-class "$EXPECTED_V2_PER_CLASS" \
  --seed "$SEED" --selection-strategy "$SELECTION_STRATEGY" \
  --output-columns-json "$OUTPUT_COLUMNS_JSON" --expected-filenames-json "$EXPECTED_FILENAMES_JSON" --legacy-filenames-json "$LEGACY_FILENAMES_JSON" \
  --manifest-filename "$MANIFEST_FILENAME" \
  --owner-map-source "$OWNER_MAP_SOURCE" --report-filename "$REPORT_FILENAME"
pass_step
