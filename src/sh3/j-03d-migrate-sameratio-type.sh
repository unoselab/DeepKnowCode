#!/usr/bin/env bash
# Standalone J-03d v7 migration of verified v6 balanced positives.
# Run once from DeepKnowCode root; does not call any other shell wrapper.
# Preserve all v6 Manifest contents and bytes. Never overwrite a destination.
set -Eeuo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
EXPERIMENT="j-03d"
VERSION="v7"
MODE="migrate-sameratio-type"
BASE="$ROOT/data/clone-detection/3pos-clones"
EXPECTED_SUBJECTS=47
LOG_DIR="$ROOT/logs/j-03d"
LOG_PREFIX="j-03d-v7-migrate-sameratio-type"
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
(( $# == 0 )) || { echo "Usage: bash src/sh3/j-03d-migrate-sameratio-type.sh" >&2; exit 2; }
[[ -d "$BASE" && ! -L "$BASE" ]] || { echo "Missing positive root: $BASE" >&2; exit 2; }
mkdir -p "$LOG_DIR"
TAG="$(date "+$LOG_TIMESTAMP_FORMAT")-$$-$(od -An -tx1 -N6 /dev/urandom | tr -d ' \n')"
LOG_FILE="$LOG_DIR/$LOG_PREFIX-$TAG.log"
( set -C; : > "$LOG_FILE" ) || { echo "Log collision: $LOG_FILE" >&2; exit 2; }
exec > >(tee -a "$LOG_FILE") 2>&1
START_TIME="$(now)"
START_EPOCH="$(date +%s)"
CURRENT_STEP="init"
CURRENT_SUBJECT="global"
finish() {
  local ec="$?"; trap - EXIT
  local status="PASS"
  (( ec == 0 )) || status="FAIL"
  echo "[$(now)] $status FINAL experiment=$EXPERIMENT version=$VERSION mode=$MODE step=$CURRENT_STEP subject=$CURRENT_SUBJECT exit=$ec start=$START_TIME end=$(now) elapsed_sec=$(($(date +%s)-START_EPOCH)) log=$LOG_FILE base=$BASE"
  exit "$ec"
}
trap finish EXIT
trap 'exit 130' INT TERM
command -v flock >/dev/null 2>&1 || { echo "flock required" >&2; exit 2; }
exec 9>"$ROOT/data/clone-detection/.j03d-positive-clones.lock"
flock -n 9 || { echo "J-03d lock busy" >&2; exit 2; }
shopt -s nullglob
SUBJECTS=("$BASE"/*__*)
(( ${#SUBJECTS[@]} == EXPECTED_SUBJECTS )) || {
  echo "Expected $EXPECTED_SUBJECTS subjects; found ${#SUBJECTS[@]}" >&2; exit 2;
}
for dir in "${SUBJECTS[@]}"; do
  [[ -d "$dir" && ! -L "$dir" ]] || { echo "Invalid subject directory $dir" >&2; exit 2; }
done
# Preflight ALL subjects before moving ANY file.
CURRENT_STEP="preflight"
echo "[$(now)] START step=preflight subjects=${#SUBJECTS[@]}"
for dir in "${SUBJECTS[@]}"; do
  CURRENT_SUBJECT="${dir##*/}"
  [[ ! -L "$dir/details" ]] || { echo "Unsafe details symlink $dir" >&2; exit 2; }
  for spec in \
      "fixed_positive_clonepairs.jsonl|sameratio-type/fixed_positive_clonepairs.jsonl" \
      "details/fixed_positive_clonepairs.jsonl|details/sameratio-type/fixed_positive_clonepairs.jsonl" \
      "details/fixed_positive_clonepairs.manifest.json|details/sameratio-type/fixed_positive_clonepairs.manifest.json"; do
    src="${spec%%|*}"; dst="${spec#*|}"
    [[ -f "$dir/$src" && ! -L "$dir/$src" ]] || { echo "Required v6 source missing $dir/$src" >&2; exit 2; }
    [[ ! -e "$dir/$dst" && ! -L "$dir/$dst" ]] || { echo "Destination exists, refusing overwrite $dir/$dst" >&2; exit 2; }
  done
  [[ ! -L "$dir/sameratio-type" && ! -L "$dir/details/sameratio-type" ]] || {
    echo "Unsafe sameratio directory symlink $dir" >&2; exit 2;
  }
done
CURRENT_SUBJECT="global"
echo "[$(now)] PASS step=preflight subjects=${#SUBJECTS[@]}"
CURRENT_STEP="move_and_hash_verify"
echo "[$(now)] START step=move_and_hash_verify"
for dir in "${SUBJECTS[@]}"; do
  CURRENT_SUBJECT="${dir##*/}"
  mkdir -p "$dir/sameratio-type" "$dir/details/sameratio-type"
  for spec in \
      "fixed_positive_clonepairs.jsonl|sameratio-type/fixed_positive_clonepairs.jsonl" \
      "details/fixed_positive_clonepairs.jsonl|details/sameratio-type/fixed_positive_clonepairs.jsonl" \
      "details/fixed_positive_clonepairs.manifest.json|details/sameratio-type/fixed_positive_clonepairs.manifest.json"; do
    src="${spec%%|*}"; dst="${spec#*|}"
    before="$(sha256sum "$dir/$src" | cut -d ' ' -f 1)"
    mv -- "$dir/$src" "$dir/$dst"
    after="$(sha256sum "$dir/$dst" | cut -d ' ' -f 1)"
    [[ "$before" == "$after" ]] || { echo "SHA256 changed during relocation $dir/$dst" >&2; exit 2; }
    echo "[$(now)] MOVE_PASS subject=$CURRENT_SUBJECT from=$src to=$dst sha256=$after"
  done
done
CURRENT_SUBJECT="global"
echo "[$(now)] PASS step=move_and_hash_verify migrated_subjects=${#SUBJECTS[@]} files=$((3*${#SUBJECTS[@]}))"
