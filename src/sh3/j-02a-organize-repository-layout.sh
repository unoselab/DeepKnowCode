#!/usr/bin/env bash
# J-02a v5: copy Java snapshots, move Python repositories without symlinks, and inventory hidden Java Git metadata.
# Standalone wrapper; does not call any other project shell wrapper.
set -Eeuo pipefail
export TZ='America/Chicago'

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
EXPERIMENT='j-02a'
VERSION='v5'
MODE="${1:---inspect}"
if [[ $# -gt 1 ]]; then echo "USAGE: bash $0 [--inspect|--apply|--verify]" >&2; exit 2; fi
case "$MODE" in --inspect|--apply|--verify) ;; *) echo "USAGE: bash $0 [--inspect|--apply|--verify]" >&2; exit 2 ;; esac

# All configuration is explicit in this Shell script.
REPOSITORY_ROOT="$ROOT/data/repositories"
PYTHON_DEST_ROOT="$REPOSITORY_ROOT/python-repos"
JAVA_DEST_ROOT="$REPOSITORY_ROOT/java-repos"
JAVA_SOURCE_ROOT="$ROOT/../2026JournalData/RQ3/java/code/Dataset-Preparation/detect_clones/NiCad/systems"
JAVA_DIR_GLOB='*-java'
# Scan the full RQ3 Java workspace, not only NiCad/systems; include nested .git directories and .git files.
JAVA_GIT_SCAN_ROOT="$ROOT/../2026JournalData/RQ3/java"
EXPECTED_JAVA_PROJECT_COUNT=47
PYTHON_PROJECTS=(
  ansible boto3 core django gensim h2ogpt horizon ironic keras keystone
  libcloud matplotlib neutron nova pandas paramiko Python salt scikit-learn
  sunnypilot swift sympy web2py werkzeug
)
EXPECTED_PYTHON_PROJECT_COUNT=24
ALLOW_CLEANUP_OF_V4_LINKS='true'  # remove only exact v4 compatibility links after validating the target
MIN_FREE_KIB_AFTER_COPY=1048576  # 1 GiB; fail before copying if insufficient
REQUIRE_PYTHON_GIT='true'
RSYNC_BIN='rsync'
RSYNC_COPY_FLAGS=(-a)
RSYNC_VERIFY_FLAGS=(-a -c -n --delete --itemize-changes)
LOG_DIR="$ROOT/logs/j-02a"
RESULTS_DIR="$ROOT/results/j-02a"
LOG_PREFIX='j-02a-v5-repository-layout'
LOG_TIMESTAMP_FORMAT='%Y-%m%d-%H%M%S'
LOG_TEE=true
LOG_CAPTURE_STDOUT=true
LOG_CAPTURE_STDERR=true

mkdir -p -- "$LOG_DIR" "$RESULTS_DIR"
STAMP="$(date +"$LOG_TIMESTAMP_FORMAT")-$$"
LOG="$(mktemp "$LOG_DIR/${LOG_PREFIX}-${STAMP}-XXXXXX.log")"
RUN="$(mktemp -d "$RESULTS_DIR/${STAMP}-repos-XXXXXX")"
REPORT="$RUN/repository-layout-report.tsv"
SUMMARY="$RUN/repository-layout-summary.txt"
GIT_INVENTORY="$RUN/java-git-metadata-inventory.tsv"
GIT_INVENTORY_SHA="$RUN/java-git-metadata-inventory.sha256"
exec > >(tee -a "$LOG") 2>&1
now() { date +'%Y-%m-%dT%H:%M:%S%:z'; }
START_SEC="$(date +%s)"
START_CHICAGO="$(now)"
CURRENT_STEP=''
STEP_START=0

finish() {
  local rc=$? status='FAIL'
  if [[ "$rc" -eq 0 ]]; then status='PASS'; fi
  echo "[$(now)] $status FINAL experiment=$EXPERIMENT version=$VERSION mode=$MODE exit=$rc elapsed_sec=$(( $(date +%s)-START_SEC )) start=$START_CHICAGO end=$(now) log=$LOG run=$RUN report=$REPORT summary=$SUMMARY"
}
failed() {
  local rc=$?
  if [[ -n "$CURRENT_STEP" ]]; then
    echo "[$(now)] FAIL step=$CURRENT_STEP elapsed_sec=$(( $(date +%s)-STEP_START )) exit=$rc" >&2
  fi
  exit "$rc"
}
trap finish EXIT
trap failed ERR
step_start() { CURRENT_STEP="$1"; STEP_START="$(date +%s)"; echo "[$(now)] START step=$CURRENT_STEP"; }
step_pass() { echo "[$(now)] PASS step=$CURRENT_STEP elapsed_sec=$(( $(date +%s)-STEP_START ))"; CURRENT_STEP=''; }
die() { echo "ERROR: $*" >&2; return 2; }
exists_path() { [[ -e "$1" || -L "$1" ]]; }

# For Git repositories, prevent accidentally using the enclosing DeepKnowCode Git repo.
check_python_git() {
  local location="$1" name="$2" resolved top
  [[ -d "$location/.git" && ! -L "$location/.git" ]] || die "Expected a standalone .git directory; linked worktree/submodule gitdir is not safe to move: $name ($location)"
  if git -C "$location" config --local --get core.worktree >/dev/null; then
    die "Python repository specifies core.worktree; manual relocation required: $name ($location)"
  fi
  resolved="$(realpath -e -- "$location")"
  top="$(git -C "$location" rev-parse --show-toplevel)"
  [[ "$(realpath -e -- "$top")" == "$resolved" ]] || die "Git worktree mismatch: $name ($top != $resolved)"
}

# Verify source and destination byte content and metadata without modifying either.
check_java_copy() {
  local source="$1" destination="$2" changes
  [[ -d "$destination" && ! -L "$destination" ]] || die "Missing/unsafe Java destination: $destination"
  changes="$($RSYNC_BIN "${RSYNC_VERIFY_FLAGS[@]}" -- "$source/" "$destination/")" || die "rsync verification failed: $destination"
  [[ -z "$changes" ]] || { printf '%s\n' "$changes" | head -n 25 >&2; die "Java snapshot mismatch: $destination"; }
}

printf '[%s] START experiment=%s version=%s mode=%s log=%s run=%s source=%s python_target=%s java_target=%s\n' \
  "$(now)" "$EXPERIMENT" "$VERSION" "$MODE" "$LOG" "$RUN" "$JAVA_SOURCE_ROOT" "$PYTHON_DEST_ROOT" "$JAVA_DEST_ROOT"

step_start validate_inputs
for cmd in "$RSYNC_BIN" find sort du df git realpath mktemp tee mv rm awk sha256sum readlink; do
  command -v "$cmd" >/dev/null || die "Missing required command: $cmd"
done
[[ "$ALLOW_CLEANUP_OF_V4_LINKS" == 'true' ]] || die 'Expected explicit v4 link cleanup policy'
[[ "$REQUIRE_PYTHON_GIT" == 'true' ]] || die 'Expected Python projects must be Git repositories'
[[ -d "$REPOSITORY_ROOT" && ! -L "$REPOSITORY_ROOT" ]] || die "Missing or unsafe repository root: $REPOSITORY_ROOT"
[[ -d "$JAVA_SOURCE_ROOT" && ! -L "$JAVA_SOURCE_ROOT" ]] || die "Missing or unsafe Java snapshot root: $JAVA_SOURCE_ROOT"
[[ -d "$JAVA_GIT_SCAN_ROOT" && ! -L "$JAVA_GIT_SCAN_ROOT" ]] || die "Missing or unsafe Java Git scan root: $JAVA_GIT_SCAN_ROOT"
for dest_root in "$PYTHON_DEST_ROOT" "$JAVA_DEST_ROOT"; do
  [[ ! -L "$dest_root" ]] || die "Destination root must not be a symlink: $dest_root"
  [[ ! -e "$dest_root" || -d "$dest_root" ]] || die "Destination root is not a directory: $dest_root"
done
[[ "${#PYTHON_PROJECTS[@]}" -eq "$EXPECTED_PYTHON_PROJECT_COUNT" ]] || die 'Python project config count mismatch'
[[ "$(realpath -e -- "$JAVA_SOURCE_ROOT")" != "$(realpath -e -- "$REPOSITORY_ROOT")" ]] || die 'Source and destination roots must differ'

mapfile -d '' -t JAVA_SOURCE_DIRS < <(find "$JAVA_SOURCE_ROOT" -mindepth 1 -maxdepth 1 -type d -name "$JAVA_DIR_GLOB" -print0 | sort -z)
[[ "${#JAVA_SOURCE_DIRS[@]}" -eq "$EXPECTED_JAVA_PROJECT_COUNT" ]] || die "Java source project count mismatch expected=$EXPECTED_JAVA_PROJECT_COUNT actual=${#JAVA_SOURCE_DIRS[@]}"
for source in "${JAVA_SOURCE_DIRS[@]}"; do
  [[ -d "$source" && ! -L "$source" ]] || die "Unsafe Java snapshot: $source"
done
step_pass

step_start preflight_python_layout
PYTHON_AT_ROOT=0
PYTHON_MOVED=0
PYTHON_V4_LINKS=0
for name in "${PYTHON_PROJECTS[@]}"; do
  old="$REPOSITORY_ROOT/$name"
  new="$PYTHON_DEST_ROOT/$name"
  if [[ -d "$old" && ! -L "$old" ]]; then
    [[ ! -e "$new" && ! -L "$new" ]] || die "Ambiguous Python locations: $old and $new"
    check_python_git "$old" "$name"
    PYTHON_AT_ROOT=$((PYTHON_AT_ROOT+1))
  elif [[ -L "$old" ]]; then
    [[ "$(readlink -- "$old")" == "python-repos/$name" ]] || die "Unexpected Python symlink (not a known v4 link): $old -> $(readlink -- "$old")"
    [[ -d "$new" && ! -L "$new" ]] || die "Missing Python target: $new"
    check_python_git "$new" "$name"
    PYTHON_MOVED=$((PYTHON_MOVED+1))
    PYTHON_V4_LINKS=$((PYTHON_V4_LINKS+1))
  elif [[ ! -e "$old" ]]; then
    [[ -d "$new" && ! -L "$new" ]] || die "Python project missing from both locations: $name"
    check_python_git "$new" "$name"
    PYTHON_MOVED=$((PYTHON_MOVED+1))
  else
    die "Unexpected Python source file: $old"
  fi
done
step_pass

step_start preflight_java_layout
JAVA_ALREADY_COPIED=0
REMAINING_COPY_KIB=0
for source in "${JAVA_SOURCE_DIRS[@]}"; do
  name="$(basename -- "$source")"
  new="$JAVA_DEST_ROOT/$name"
  if exists_path "$new"; then
    [[ -d "$new" && ! -L "$new" ]] || die "Non-directory or symlink blocks Java destination: $new"
    JAVA_ALREADY_COPIED=$((JAVA_ALREADY_COPIED+1))
  else
    kib="$(du -sk -- "$source" | awk '{print $1}')"
    [[ "$kib" =~ ^[0-9]+$ ]] || die "Invalid source size for $name"
    REMAINING_COPY_KIB=$((REMAINING_COPY_KIB+kib))
  fi
done
FREE_KIB="$(df -Pk -- "$REPOSITORY_ROOT" | awk 'NR==2 {print $4}')"
[[ "$FREE_KIB" =~ ^[0-9]+$ ]] || die 'Unable to determine free disk space'
{
  echo "timestamp_chicago=$(now)"
  echo "experiment=$EXPERIMENT version=$VERSION mode=$MODE"
  echo "java_source_snapshot_dir=$JAVA_SOURCE_ROOT"
  echo "java_source_count=${#JAVA_SOURCE_DIRS[@]}"
  echo "java_copied_count=$JAVA_ALREADY_COPIED"
  echo "python_expected_count=${#PYTHON_PROJECTS[@]}"
  echo "python_still_at_root=$PYTHON_AT_ROOT"
  echo "python_moved_or_partially_moved=$PYTHON_MOVED"
  echo "python_v4_links_to_remove_on_apply=$PYTHON_V4_LINKS"
  echo "remaining_java_copy_kib=$REMAINING_COPY_KIB"
  echo "available_disk_kib=$FREE_KIB"
  echo "min_free_after_copy_kib=$MIN_FREE_KIB_AFTER_COPY"
  echo "java_origin_type=nicad_source_snapshot_not_proven_git_repository"
  echo "python_compatibility_symlink_policy=NO_SYMLINKS"
  echo "java_git_scan_root=$JAVA_GIT_SCAN_ROOT"
} | tee "$SUMMARY"
if [[ "$MODE" == '--apply' && "$FREE_KIB" -lt "$((REMAINING_COPY_KIB+MIN_FREE_KIB_AFTER_COPY))" ]]; then
  die "Insufficient free space: available=${FREE_KIB}KiB required=$((REMAINING_COPY_KIB+MIN_FREE_KIB_AFTER_COPY))KiB"
fi
step_pass

step_start discover_java_git_metadata
# Read-only inventory: source snapshots plus other Java workspaces under RQ3/java.
# -prune prevents recursively scanning Git object databases; -P find semantics do not follow symlinks.
GIT_MARKERS_RAW="$RUN/java-git-markers.nul"
find "$JAVA_GIT_SCAN_ROOT" -name .git -prune -print0 > "$GIT_MARKERS_RAW"
mapfile -d '' -t JAVA_GIT_MARKERS < <(sort -z -- "$GIT_MARKERS_RAW")
printf 'scope\tproject\tgit_marker_type\tgit_marker_path\tgit_top_level\tobserved_head\tstatus\n' > "$GIT_INVENTORY"
JAVA_GIT_ROOT_COUNT=0
JAVA_GIT_NESTED_COUNT=0
JAVA_OTHER_GIT_COUNT=0
for marker in "${JAVA_GIT_MARKERS[@]}"; do
  scope='OTHER_RQ3_JAVA'
  project='-'
  git_parent="$(dirname -- "$marker")"
  for source in "${JAVA_SOURCE_DIRS[@]}"; do
    if [[ "$marker" == "$source/.git" ]]; then
      scope='JAVA_SNAPSHOT_ROOT'; project="$(basename -- "$source")"
      JAVA_GIT_ROOT_COUNT=$((JAVA_GIT_ROOT_COUNT+1)); break
    elif [[ "$marker" == "$source/"* ]]; then
      scope='JAVA_SNAPSHOT_NESTED'; project="$(basename -- "$source")"
      JAVA_GIT_NESTED_COUNT=$((JAVA_GIT_NESTED_COUNT+1)); break
    fi
  done
  if [[ "$scope" == 'OTHER_RQ3_JAVA' ]]; then
    JAVA_OTHER_GIT_COUNT=$((JAVA_OTHER_GIT_COUNT+1))
  fi
  if [[ -L "$marker" ]]; then
    git_type='SYMLINK'; observed_head='-'; git_top='-'; git_status='UNSAFE_GIT_MARKER_LINK'
  elif [[ -d "$marker" ]]; then
    git_type='DIRECTORY'; observed_head='-'; git_top='-'; git_status='INVALID_OR_UNVERIFIED'
  elif [[ -f "$marker" ]]; then
    git_type='FILE'; observed_head='-'; git_top='-'; git_status='INVALID_OR_UNVERIFIED'
  else
    git_type='OTHER'; observed_head='-'; git_top='-'; git_status='INVALID_OR_UNVERIFIED'
  fi
  if [[ "$git_type" == 'DIRECTORY' || "$git_type" == 'FILE' ]]; then
    if observed_head="$(git -C "$git_parent" rev-parse --verify HEAD 2>/dev/null)" && git_top="$(git -C "$git_parent" rev-parse --show-toplevel 2>/dev/null)"; then
      if [[ "$(realpath -e -- "$git_top")" == "$(realpath -e -- "$git_parent")" ]]; then
        git_status='GIT_METADATA_PRESENT_HEAD_OBSERVED_NOT_DATASET_PINNED'
      else
        git_status='GIT_TOPLEVEL_NOT_MARKER_PARENT'
      fi
    else
      observed_head='-'; git_top='-'; git_status='NO_USABLE_GIT_HEAD'
    fi
  fi
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$scope" "$project" "$git_type" "$marker" "$git_top" "$observed_head" "$git_status" >> "$GIT_INVENTORY"
done
sha256sum -- "$GIT_INVENTORY" | tee "$GIT_INVENTORY_SHA"
{
  echo "java_git_markers_total=${#JAVA_GIT_MARKERS[@]}"
  echo "java_git_markers_at_snapshot_roots=$JAVA_GIT_ROOT_COUNT"
  echo "java_git_markers_nested_in_snapshots=$JAVA_GIT_NESTED_COUNT"
  echo "java_git_markers_elsewhere_in_rq3_java=$JAVA_OTHER_GIT_COUNT"
  echo "java_git_inventory=$GIT_INVENTORY"
  echo "java_git_inventory_sha256=$(awk '{print $1}' "$GIT_INVENTORY_SHA")"
} | tee -a "$SUMMARY"
echo "[$(now)] GIT_SCAN source_root=$JAVA_GIT_SCAN_ROOT markers=${#JAVA_GIT_MARKERS[@]} snapshot_roots=$JAVA_GIT_ROOT_COUNT snapshot_nested=$JAVA_GIT_NESTED_COUNT other=$JAVA_OTHER_GIT_COUNT inventory=$GIT_INVENTORY"
step_pass

if [[ "$MODE" == '--inspect' ]]; then
  echo "[$(now)] INSPECT_PASS Java source snapshots=${#JAVA_SOURCE_DIRS[@]} Python Git repositories=${#PYTHON_PROJECTS[@]} java_git_markers=${#JAVA_GIT_MARKERS[@]}"
  exit 0
fi

if [[ "$MODE" == '--apply' ]]; then
  step_start copy_java_snapshots
  mkdir -p -- "$JAVA_DEST_ROOT" "$PYTHON_DEST_ROOT"
  for source in "${JAVA_SOURCE_DIRS[@]}"; do
    name="$(basename -- "$source")"
    destination="$JAVA_DEST_ROOT/$name"
    if exists_path "$destination"; then
      echo "[$(now)] CHECK existing_java_snapshot=$name"
      check_java_copy "$source" "$destination"
      continue
    fi
    # Stage in the destination filesystem; rename only after full content verification.
    staging="$(mktemp -d "$JAVA_DEST_ROOT/.${name}.${STAMP}.stage.XXXXXX")"
    echo "[$(now)] COPY java_snapshot=$name staging=$staging"
    "$RSYNC_BIN" "${RSYNC_COPY_FLAGS[@]}" -- "$source/" "$staging/"
    check_java_copy "$source" "$staging"
    [[ ! -e "$destination" && ! -L "$destination" ]] || die "Java target appeared during copy: $destination"
    mv -T -- "$staging" "$destination"
    echo "[$(now)] PASS java_snapshot=$name destination=$destination"
  done
  step_pass

  step_start move_python_git_repositories
  for name in "${PYTHON_PROJECTS[@]}"; do
    old="$REPOSITORY_ROOT/$name"
    new="$PYTHON_DEST_ROOT/$name"
    if [[ -d "$old" && ! -L "$old" ]]; then
      [[ ! -e "$new" && ! -L "$new" ]] || die "Cannot move: destination exists: $new"
      check_python_git "$old" "$name"
      mv -T -- "$old" "$new"
      echo "[$(now)] MOVE python_git_repository=$name destination=$new"
    fi
    [[ -d "$new" && ! -L "$new" ]] || die "Python destination missing after move: $new"
    check_python_git "$new" "$name"
    if [[ -L "$old" ]]; then
      [[ "$(readlink -- "$old")" == "python-repos/$name" ]] || die "Refusing to remove unknown symlink: $old"
      check_python_git "$new" "$name"
      rm -- "$old"
      echo "[$(now)] REMOVE_EXACT_V4_LINK python_legacy=$old"
    fi
    [[ ! -e "$old" && ! -L "$old" ]] || die "Python legacy path must be absent: $old"
  done
  step_pass
fi

step_start verify_java_snapshots
printf 'kind\tproject\toriginal_path\tstaged_path\trevision\tverification\n' > "$REPORT"
for source in "${JAVA_SOURCE_DIRS[@]}"; do
  name="$(basename -- "$source")"
  new="$JAVA_DEST_ROOT/$name"
  check_java_copy "$source" "$new"
  printf 'java_nicad_snapshot\t%s\t%s\t%s\t%s\t%s\n' \
    "$name" "$source" "$new" 'UNKNOWN_NOT_GIT_PINNED' 'RSYNC_CHECKSUM_VERIFIED' >> "$REPORT"
done
step_pass

step_start verify_python_git_repositories
for name in "${PYTHON_PROJECTS[@]}"; do
  old="$REPOSITORY_ROOT/$name"
  new="$PYTHON_DEST_ROOT/$name"
  [[ ! -e "$old" && ! -L "$old" ]] || die "Python legacy path must be absent (no symlinks): $old"
  check_python_git "$new" "$name"
  revision="$(git -C "$new" rev-parse HEAD)"
  printf 'python_git_repo\t%s\t%s\t%s\t%s\t%s\n' \
    "$name" "$old" "$new" "$revision" 'MOVED_NO_LEGACY_SYMLINK' >> "$REPORT"
done
step_pass

step_start finalize_manifest
sha256sum -- "$REPORT" | tee "$RUN/repository-layout-report.sha256"
printf '[%s] VERIFIED java_snapshots=%s python_git_repositories=%s report=%s\n' \
  "$(now)" "${#JAVA_SOURCE_DIRS[@]}" "${#PYTHON_PROJECTS[@]}" "$REPORT"
step_pass
