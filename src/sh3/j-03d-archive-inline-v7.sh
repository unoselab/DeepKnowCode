#!/usr/bin/env bash
# Standalone J-03d v9 archival of v7 root inline-code positives; no overwrite.
set -Eeuo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
EXPERIMENT="j-03d"
VERSION="v9"
MODE="archive-inline-v7"
BASE="$ROOT/data/clone-detection/java/3pos-clones"
EXPECTED_SUBJECTS=47
LOG_DIR="$ROOT/logs/j-03d"
LOG_PREFIX="j-03d-v9-archive-inline-v7"
LOG_TIMESTAMP_FORMAT="%Y%m%d-%H%M%S-%6N"
LOG_TIME_ZONE="America/Chicago"
LOG_TEE_APPEND="true"
LOG_CAPTURE_STDOUT="true"
LOG_CAPTURE_STDERR="true"
LOG_STEP_TIMING="true"
LOG_SHOW_TERMINAL="true"
PYTHON_BIN="python3"
export TZ="$LOG_TIME_ZONE"
export LC_ALL=C
now() { date '+%Y-%m-%dT%H:%M:%S%:z'; }
(( $# == 0 )) || { echo "Usage: bash src/sh3/j-03d-archive-inline-v7.sh" >&2; exit 2; }
[[ -d "$BASE" && ! -L "$BASE" ]] || { echo "Missing positive root $BASE" >&2; exit 2; }
mkdir -p "$LOG_DIR"
TAG="$(date "+$LOG_TIMESTAMP_FORMAT")-$$-$(od -An -tx1 -N6 /dev/urandom | tr -d ' \n')"
LOG_FILE="$LOG_DIR/$LOG_PREFIX-$TAG.log"
( set -C; : > "$LOG_FILE" ) || { echo "Log collision $LOG_FILE" >&2; exit 2; }
exec > >(tee -a "$LOG_FILE") 2>&1
START_TIME="$(now)"; START_EPOCH="$(date +%s)"
CURRENT_STEP="preflight"; CURRENT_SUBJECT="global"
finish() {
  local ec="$?"; trap - EXIT
  local status=PASS
  (( ec == 0 )) || status=FAIL
  echo "[$(now)] $status FINAL experiment=$EXPERIMENT version=$VERSION mode=$MODE step=$CURRENT_STEP subject=$CURRENT_SUBJECT exit=$ec start=$START_TIME end=$(now) elapsed_sec=$(($(date +%s)-START_EPOCH)) log=$LOG_FILE"
  exit "$ec"
}
trap finish EXIT
trap 'exit 130' INT TERM
command -v flock >/dev/null || { echo "flock is required" >&2; exit 2; }
command -v "$PYTHON_BIN" >/dev/null || { echo "Python is required" >&2; exit 2; }
exec 9>"$ROOT/data/clone-detection/.j03d-positive-clones.lock"
flock -n 9 || { echo "J-03d lock busy" >&2; exit 2; }
shopt -s nullglob
SUBJECTS=("$BASE"/*__*)
(( ${#SUBJECTS[@]} == EXPECTED_SUBJECTS )) || { echo "Expected $EXPECTED_SUBJECTS subject directories, got ${#SUBJECTS[@]}" >&2; exit 2; }
MOVE_COUNT=0
printf '[%s] START step=preflight subjects=%d\n' "$(now)" "${#SUBJECTS[@]}"
for dir in "${SUBJECTS[@]}"; do
  CURRENT_SUBJECT="${dir##*/}"
  [[ -d "$dir" && ! -L "$dir" && ! -L "$dir/details" ]] || { echo "Unsafe subject $dir" >&2; exit 2; }
  pub="$dir/fixed_positive_clonepairs.jsonl"
  det="$dir/details/fixed_positive_clonepairs.jsonl"
  man="$dir/details/fixed_positive_clonepairs.manifest.json"
  if [[ ! -e "$pub" && ! -e "$det" && ! -e "$man" ]]; then
    echo "[$(now)] SKIP subject=$CURRENT_SUBJECT reason=no_existing_root_files"
    continue
  fi
  for path in "$pub" "$det" "$man"; do
    [[ -f "$path" && ! -L "$path" ]] || { echo "Incomplete v7 root artifacts: $path" >&2; exit 2; }
  done
  for path in "$dir/legacy-v7" "$dir/details/legacy-v7" \
      "$dir/legacy-v7/fixed_positive_clonepairs.jsonl" \
      "$dir/details/legacy-v7/fixed_positive_clonepairs.jsonl" \
      "$dir/details/legacy-v7/fixed_positive_clonepairs.manifest.json"; do
    [[ ! -e "$path" && ! -L "$path" ]] || { echo "Destination already exists: $path" >&2; exit 2; }
  done
  "$PYTHON_BIN" - "$pub" "$det" "$man" <<'PY'
import hashlib,json,sys
from pathlib import Path
pub,det,man=map(Path,sys.argv[1:])
m=json.loads(man.read_text())
if m.get('version')!='v7' or m.get('effective_configuration',{}).get('output_layout')!='root':
    raise SystemExit('Only v7 root artifacts may be archived: '+str(man))
for path,key in ((pub,'fixed_positive_clonepairs.jsonl'),(det,'details/fixed_positive_clonepairs.jsonl')):
    h=hashlib.file_digest(path.open('rb'),'sha256').hexdigest()
    if h!=m['files_sha256'][key]:raise SystemExit('SHA256 mismatch: '+str(path))
PY
  MOVE_COUNT=$((MOVE_COUNT+1))
done
printf '[%s] PASS step=preflight prepared_subjects=%d\n' "$(now)" "$MOVE_COUNT"
CURRENT_STEP="archive"
for dir in "${SUBJECTS[@]}"; do
  CURRENT_SUBJECT="${dir##*/}"
  [[ -f "$dir/fixed_positive_clonepairs.jsonl" ]] || continue
  echo "[$(now)] START step=archive subject=$CURRENT_SUBJECT"
  mkdir "$dir/legacy-v7" "$dir/details/legacy-v7"
  mv -n -- "$dir/fixed_positive_clonepairs.jsonl" "$dir/legacy-v7/fixed_positive_clonepairs.jsonl"
  mv -n -- "$dir/details/fixed_positive_clonepairs.jsonl" "$dir/details/legacy-v7/fixed_positive_clonepairs.jsonl"
  mv -n -- "$dir/details/fixed_positive_clonepairs.manifest.json" "$dir/details/legacy-v7/fixed_positive_clonepairs.manifest.json"
  echo "[$(now)] PASS step=archive subject=$CURRENT_SUBJECT preserved_files=3"
done
CURRENT_SUBJECT="global"
echo "[$(now)] PASS step=archive completed_subjects=$MOVE_COUNT"
