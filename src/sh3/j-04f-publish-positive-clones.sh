#!/usr/bin/env bash
# DeepKnowCode J-04f v1 -- Publish verified J-04e fixed positives without reclassification.
# Run from DeepKnowCode repository root, in this exact order:
# 1. Inspect immutable J-04e ALL/VERIFY evidence and every subject artifact:
# bash src/sh3/j-04f-publish-positive-clones.sh --inspect
# 2. Dry-run exact publication plan (no change to final dataset):
# bash src/sh3/j-04f-publish-positive-clones.sh --dry-run
# 3. Copy verified data to 3pos-clones (fails if destination nonempty):
# bash src/sh3/j-04f-publish-positive-clones.sh --copy-result
# 4. Independently verify final published files against J-04e SHA256:
# bash src/sh3/j-04f-publish-positive-clones.sh --verify
# No --all or --smoke: this is a publication-only stage.
# No deletion, overwriting, or changes to J-04e, J-04d or existing published data.
set -Eeuo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
EXPERIMENT='j-04f'
VERSION='v1'
SOURCE_EXPERIMENT='j-04e'
SOURCE_VERSION='v1'
SOURCE_ALL_RUN_ID='20261121-002935-080151599-pid2629742-all-4eeef2546b'
SOURCE_VERIFY_RUN_ID='20261121-003001-074277851-pid2629872-verify-68bfc694d4'
SOURCE_ALL_ROOT="$ROOT/results/j-04e/$SOURCE_ALL_RUN_ID"
SOURCE_VERIFY_ROOT="$ROOT/results/j-04e/$SOURCE_VERIFY_RUN_ID"
RESULT_ROOT="$ROOT/results/j-04f"
PUBLISH_ROOT="$ROOT/data/clone-detection/python/3pos-clones"
PYTHON_BIN="$HOME/miniconda3/envs/ast-py314/bin/python"
PYTHON_SCRIPT="$ROOT/src/py3/j-04f-publish-positive-clones.py"
EXPECTED_PYTHON_VERSION='3.14'
EXPECTED_SUBJECTS=24
EXPECTED_PAIRS=11141
EXPECTED_FRAGMENTS=11959
EXPECTED_FIXED_POSITIVE=8633
EXPECTED_DUPLICATES=44
POSITIVE_LABEL=1
LOG_DIR="$ROOT/logs/j-04f"
LOG_PREFIX='j-04f-v1-publication'
LOG_TIMESTAMP_FORMAT='%Y%m%d-%H%M%S-%N'
LOG_TIMEZONE='America/Chicago'
LOG_CAPTURE_STDOUT='true'
LOG_CAPTURE_STDERR='true'
LOG_TEE_APPEND='false'
LOG_STEP_LOGGING='true'
LOG_PRESERVE_EXIT_CODE='true'
export TZ="$LOG_TIMEZONE"
export LC_ALL=C
usage() { echo 'Usage: bash src/sh3/j-04f-publish-positive-clones.sh --inspect|--dry-run|--copy-result|--verify' >&2; exit 2; }
(( $# == 1 )) || usage
MODE="${1#--}"
case "$1" in --inspect|--dry-run|--copy-result|--verify) ;; *) usage ;; esac
command -v tee >/dev/null || { echo 'FAIL tee unavailable' >&2; exit 2; }
[[ -x "$PYTHON_BIN" && -f "$PYTHON_SCRIPT" && ! -L "$PYTHON_SCRIPT" ]] || { echo 'FAIL Python runtime/script unavailable' >&2; exit 2; }
mkdir -p "$LOG_DIR" "$RESULT_ROOT"
RUN_ID="$(date "+$LOG_TIMESTAMP_FORMAT")-pid$$-${MODE}-$(od -An -tx1 -N5 /dev/urandom | tr -d ' \n')"
RUN_DIR="$RESULT_ROOT/$RUN_ID"
LOG_FILE="$LOG_DIR/$LOG_PREFIX-$RUN_ID.log"
( set -C; : > "$LOG_FILE" ) || { echo 'FAIL log file already exists' >&2; exit 2; }
exec > >(tee "$LOG_FILE") 2>&1
now() { date '+%Y-%m-%dT%H:%M:%S%:z'; }
START_TIME="$(now)"; START_EPOCH="$(date +%s)"; STEP='preflight'; STEP_EPOCH="$START_EPOCH"
finish() {
  local rc="$?" status='FAIL'; trap - EXIT
  if (( rc == 0 )); then status='PASS'; else echo "[$(now)] FAIL step=$STEP elapsed_sec=$(($(date +%s)-STEP_EPOCH)) exit=$rc"; fi
  echo "[$(now)] FINAL experiment=$EXPERIMENT version=$VERSION mode=$MODE status=$status exit=$rc start=$START_TIME end=$(now) elapsed_sec=$(($(date +%s)-START_EPOCH)) log=$LOG_FILE result=$RUN_DIR publish=$PUBLISH_ROOT"
  exit "$rc"
}
trap finish EXIT
trap 'exit 130' INT TERM
step_start() { STEP="$1"; STEP_EPOCH="$(date +%s)"; echo "[$(now)] START step=$STEP"; }
step_pass() { echo "[$(now)] PASS step=$STEP elapsed_sec=$(($(date +%s)-STEP_EPOCH))"; }
step_start preflight
"$PYTHON_BIN" -c 'import sys; assert f"{sys.version_info.major}.{sys.version_info.minor}" == sys.argv[1]' "$EXPECTED_PYTHON_VERSION"
[[ -d "$SOURCE_ALL_ROOT" && -d "$SOURCE_VERIFY_ROOT" ]] || { echo 'FAIL pinned upstream not found'; exit 2; }
step_pass
echo "[$(now)] RUN experiment=$EXPERIMENT version=$VERSION mode=$MODE run_id=$RUN_ID timezone=$LOG_TIMEZONE log=$LOG_FILE result=$RUN_DIR"
echo "[$(now)] CONFIG source_all=$SOURCE_ALL_ROOT source_verify=$SOURCE_VERIFY_ROOT publish=$PUBLISH_ROOT subjects=$EXPECTED_SUBJECTS candidates=$EXPECTED_PAIRS fragments=$EXPECTED_FRAGMENTS fixed_positive=$EXPECTED_FIXED_POSITIVE duplicates=$EXPECTED_DUPLICATES python=$PYTHON_BIN"
step_start "$MODE"
"$PYTHON_BIN" -u "$PYTHON_SCRIPT" \
  --experiment "$EXPERIMENT" --version "$VERSION" \
  --source-experiment "$SOURCE_EXPERIMENT" --source-version "$SOURCE_VERSION" \
  --source-all-run-id "$SOURCE_ALL_RUN_ID" --source-verify-run-id "$SOURCE_VERIFY_RUN_ID" \
  --source-all-root "$SOURCE_ALL_ROOT" --source-verify-root "$SOURCE_VERIFY_ROOT" \
  --root "$ROOT" --result-root "$RESULT_ROOT" --publish-root "$PUBLISH_ROOT" \
  --run-dir "$RUN_DIR" --run-id "$RUN_ID" --mode "$MODE" \
  --timezone "$LOG_TIMEZONE" --expected-python-version "$EXPECTED_PYTHON_VERSION" \
  --expected-subjects "$EXPECTED_SUBJECTS" --expected-pairs "$EXPECTED_PAIRS" \
  --expected-fragments "$EXPECTED_FRAGMENTS" \
  --expected-fixed-positive "$EXPECTED_FIXED_POSITIVE" \
  --expected-duplicates "$EXPECTED_DUPLICATES" --positive-label "$POSITIVE_LABEL"
step_pass
