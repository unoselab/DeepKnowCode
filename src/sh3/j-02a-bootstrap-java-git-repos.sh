#!/usr/bin/env bash
# J-02a v7 - authentic public GitHub clones for the 47 NiCad Java snapshots.
# All experiment-specific configuration is defined in this standalone shell script.
set -Eeuo pipefail
export TZ='America/Chicago'
export GIT_TERMINAL_PROMPT=0
export GIT_LFS_SKIP_SMUDGE=1

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
EXPERIMENT='j-02a'
VERSION='v7'
GITHUB_BASE='https://github.com'
MODE="${1:-}"
EXPECTED_JAVA_REPOS=47
EXPECTED_PYTHON_REPOS=24
REPOSITORY_ROOT="$ROOT/data/repositories"
LIVE_DIR="$REPOSITORY_ROOT/java-repos"
STAGING_DIR="$REPOSITORY_ROOT/java-git-repos"
SNAPSHOT_ARCHIVE_DIR="$REPOSITORY_ROOT/java-snapshots"
PYTHON_DIR="$REPOSITORY_ROOT/python-repos"
LOG_DIR="$ROOT/logs/j-02a"
RESULTS_DIR="$ROOT/results/j-02a"
LOG_PREFIX='j-02a-v7-java-github'
LOG_TIMESTAMP_FORMAT='%Y-%m%d-%H%M%S'
LOG_TEE=true
LOG_CAPTURE_STDOUT=true
LOG_CAPTURE_STDERR=true
# Reproducible, small downloads; '1' means a genuine but shallow Git repository.
# To retain the entire accessible branch history, set CLONE_DEPTH='0' in Shell.
CLONE_DEPTH='1'
CLONE_SINGLE_BRANCH=true
CLONE_NO_TAGS=true
CLONE_FILTER='blob:none'
GIT_TIMEOUT_SECONDS=7200
MIN_FREE_KIB=1048576
REQUIRE_CLEAN_GIT_WORKTREE=true
REQUIRE_JAVA_FILES=true

# Snapshot directory => candidate upstream. The remote is checked online via
# 'git ls-remote' and its actual origin + HEAD are recorded; this does not prove
# correspondence to the historical NiCad snapshot.
REPO_SPECS=(
  'activemq-java|apache/activemq'
  'ant-ivy-java|apache/ant-ivy'
  'ant-java|apache/ant'
  'argouml-java|argouml-tigris-org/argouml'
  'camel-java|apache/camel'
  'cassandra-java|apache/cassandra'
  'cayenne-java|apache/cayenne'
  'commons-lang-java|apache/commons-lang'
  'commons-math-java|apache/commons-math'
  'cxf-java|apache/cxf'
  'derby-java|apache/derby'
  'dubbo-java|apache/dubbo'
  'elasticsearch-java|elastic/elasticsearch'
  'flink-java|apache/flink'
  'groovy-java|apache/groovy'
  'gson-java|google/gson'
  'hadoop-java|apache/hadoop'
  'hbase-java|apache/hbase'
  'hive-java|apache/hive'
  'ignite-java|apache/ignite'
  'jackrabbit-java|apache/jackrabbit'
  'jEdit-java|jEdit-editor/jEdit'
  'jfreechart-java|jfree/jfreechart'
  'jhotdraw7-java|wrandelshofer/jhotdraw'
  'jmeter-java|apache/jmeter'
  'joda-time-java|JodaOrg/joda-time'
  'jruby-java|jruby/jruby'
  'kafka-java|apache/kafka'
  'libgdx-java|libgdx/libgdx'
  'logging-log4j2-java|apache/logging-log4j2'
  'lucene-java|apache/lucene'
  'maven-java|apache/maven'
  'mockito-java|mockito/mockito'
  'netty-java|netty/netty'
  'openjpa-java|apache/openjpa'
  'pdfbox-java|apache/pdfbox'
  'pig-java|apache/pig'
  'poi-java|apache/poi'
  'presto-java|prestodb/presto'
  'RxJava-java|ReactiveX/RxJava'
  'storm-java|apache/storm'
  'synapse-java|apache/synapse'
  'tika-java|apache/tika'
  'tomcat-java|apache/tomcat'
  'wicket-java|apache/wicket'
  'xerces-j-java|apache/xerces-j'
  'zookeeper-java|apache/zookeeper'
)

# Source snapshots retain their historic NiCad names; actual Git clones use
# lowercase GitHub owner__repo, never the NiCad '-java' suffix.
repo_label() {
  local name="${1,,}"
  [[ "$name" =~ ^[a-z0-9_.-]+/[a-z0-9_.-]+$ ]] || die "Invalid GitHub slug: $1"
  printf '%s__%s' "${name%%/*}" "${name#*/}"
}
# Python source dirs were already moved to python-repos by v5. Derive names
# from the actual origin remote; do not guess owners from legacy directories.
python_inventory() {
  local dir name url slug owner repo label sha n=0
  printf 'old_directory\tnew_directory\torigin\thead\n' > "$PYTHON_INVENTORY"
  declare -A seen_python=()
  for dir in "$PYTHON_DIR"/*; do
    [[ -d "$dir" && ! -L "$dir" ]] || continue
    name="${dir##*/}"
    [[ -d "$dir/.git" || -f "$dir/.git" ]] || die "Python directory lacks Git metadata: $dir"
    root_match "$dir" || die "Python Git worktree root mismatch: $dir"
    url="$(git -C "$dir" remote get-url origin)" || die "Missing Python origin: $dir"
    # Support normal GitHub HTTPS and SSH remotes only.
    slug=''
    if [[ "$url" =~ ^https://github[.]com/([^/]+)/([^/]+)$ ]]; then
      slug="${BASH_REMATCH[1]}/${BASH_REMATCH[2]}"
    elif [[ "$url" =~ ^git@github[.]com:([^/]+)/([^/]+)$ ]]; then
      slug="${BASH_REMATCH[1]}/${BASH_REMATCH[2]}"
    else
      die "Python origin is not supported GitHub URL: $name => $url"
    fi
    slug="${slug%.git}"
    label="$(repo_label "$slug")"
    [[ ! -v 'seen_python[$label]' ]] || die "Duplicate Python GitHub label: $label"
    seen_python[$label]=1
    sha="$(git -C "$dir" rev-parse --verify HEAD^{commit})" || die "Invalid Python HEAD: $dir"
    printf '%s\t%s\t%s\t%s\n' "$name" "$label" "$url" "$sha" >> "$PYTHON_INVENTORY"
    ((n+=1))
  done
  [[ "$n" -eq "$EXPECTED_PYTHON_REPOS" ]] || die "Python count mismatch: $n expected $EXPECTED_PYTHON_REPOS"
  sha256sum "$PYTHON_INVENTORY"
}
rename_python() {
  local old label origin sha src dest
  # Preflight full inventory before first move.
  python_inventory
  while IFS=$'\t' read -r old label origin sha; do
    [[ "$old" != old_directory ]] || continue
    src="$PYTHON_DIR/$old"; dest="$PYTHON_DIR/$label"
    [[ "$src" == "$dest" ]] && continue
    [[ ! -e "$dest" && ! -L "$dest" ]] || die "Python destination occupied, refusing overwrite: $dest"
  done < "$PYTHON_INVENTORY"
  while IFS=$'\t' read -r old label origin sha; do
    [[ "$old" != old_directory ]] || continue
    src="$PYTHON_DIR/$old"; dest="$PYTHON_DIR/$label"
    [[ "$src" == "$dest" ]] && continue
    echo "[$(now)] START step=rename_python old=$old new=$label"
    mv -T -- "$src" "$dest"
    [[ "$(git -C "$dest" rev-parse HEAD)" == "$sha" ]] || die "Python HEAD changed after move: $dest"
    echo "[$(now)] PASS step=rename_python name=$label"
  done < "$PYTHON_INVENTORY"
  python_inventory
}
usage() {
  echo "USAGE: bash $0 {--inspect|--probe|--clone|--verify|--activate|--verify-active|--rename-python|--verify-python}" >&2
  echo '  --inspect: local paths only, no downloads or changes to repositories' >&2
  echo '  --probe:   online git ls-remote for all configured upstream URLs' >&2
  echo '  --clone:   fetch authentic repos into java-git-repos, leave snapshots untouched' >&2
  echo '  --verify:  verify all 47 staged Git clones and record HEAD + tree SHA' >&2
  echo '  --activate: move NiCad snapshots to java-snapshots, then promote clones into java-repos' >&2
  echo '  --verify-active: verify all live Git repos and archived NiCad snapshots'
  echo '  --rename-python: rename Python Git repos to lowercased owner__project (no symlinks)'
  echo '  --verify-python: verify and inventory Python Git repos and owners' >&2
}
case "$MODE" in
  --inspect|--probe|--clone|--verify|--activate|--verify-active|--rename-python|--verify-python) ;;
  *) usage; exit 2 ;;
esac
if [[ $# -ne 1 ]]; then usage; exit 2; fi
mkdir -p -- "$LOG_DIR" "$RESULTS_DIR"
STAMP="$(date +"$LOG_TIMESTAMP_FORMAT")-$$"
LOG="$(mktemp "$LOG_DIR/${LOG_PREFIX}-${STAMP}-XXXXXX.log")"
RUN="$(mktemp -d "$RESULTS_DIR/${STAMP}-git-XXXXXX")"
MANIFEST="$RUN/java-github-inventory.tsv"
PROBE="$RUN/java-github-probe.tsv"
PYTHON_INVENTORY="$RUN/python-github-inventory.tsv"
exec > >(tee -a "$LOG") 2>&1
now() { date +'%Y-%m-%dT%H:%M:%S%:z'; }
START_SEC="$(date +%s)"
START_TIME="$(now)"
CURRENT_STEP=''
STEP_START=0
finish() {
  local rc=$? status='FAIL'
  if [[ "$rc" -eq 0 ]]; then status='PASS'; fi
  echo "[$(now)] $status FINAL experiment=$EXPERIMENT version=$VERSION mode=$MODE exit=$rc elapsed_sec=$(( $(date +%s)-START_SEC )) start=$START_TIME end=$(now) log=$LOG run=$RUN manifest=$MANIFEST probe=$PROBE"
}
failed() {
  local rc=$?
  if [[ -n "$CURRENT_STEP" ]]; then
    echo "[$(now)] FAIL step=$CURRENT_STEP elapsed_sec=$(( $(date +%s)-STEP_START )) exit=$rc"
  fi
  exit "$rc"
}
trap finish EXIT
trap failed ERR
start_step() { CURRENT_STEP="$1"; STEP_START="$(date +%s)"; echo "[$(now)] START step=$CURRENT_STEP"; }
pass_step() { echo "[$(now)] PASS step=$CURRENT_STEP elapsed_sec=$(( $(date +%s)-STEP_START ))"; CURRENT_STEP=''; }
die() { echo "[$(now)] ERROR $*" >&2; return 2; }
is_snapshot() { [[ -d "$1" && ! -L "$1" && ! -e "$1/.git" && ! -L "$1/.git" ]]; }
root_match() {
  local top
  top="$(git -C "$1" rev-parse --show-toplevel 2>/dev/null)" || return 1
  [[ "$(cd "$1" && pwd -P)" == "$(cd "$top" && pwd -P)" ]]
}
verify_repo() {
  local dir="$1" url="$2" project="$3" origin sha clean found
  [[ -d "$dir" && ! -L "$dir" && -d "$dir/.git" ]] || die "Missing authentic .git directory: $dir"
  root_match "$dir" || die "Git top-level mismatch: $dir"
  origin="$(git -C "$dir" remote get-url origin)" || die "Missing origin: $dir"
  [[ "$origin" == "$url" ]] || die "Wrong origin for $project: actual=$origin expected=$url"
  sha="$(git -C "$dir" rev-parse --verify HEAD^{commit})" || die "No valid commit for $project"
  [[ "$sha" =~ ^[0-9a-f]{40,64}$ ]] || die "Invalid HEAD sha for $project: $sha"
  if [[ "$REQUIRE_CLEAN_GIT_WORKTREE" == true ]]; then
    clean="$(git -C "$dir" status --porcelain --untracked-files=normal)" || die "git status failed: $dir"
    [[ -z "$clean" ]] || die "Non-clean worktree: $dir"
  fi
  if [[ "$REQUIRE_JAVA_FILES" == true ]]; then
    # 'git ls-files' also works with partial and shallow clones.
    found="$(git -C "$dir" ls-files -- '*.java' | sed -n '1p')" || die "Cannot list files for $project"
    [[ -n "$found" ]] || die "No tracked Java files: $project"
  fi
}
check_specs() {
  local row project gh url
  [[ "$CLONE_DEPTH" =~ ^[0-9]+$ ]] || die 'CLONE_DEPTH must be a nonnegative integer'
  [[ "$MIN_FREE_KIB" =~ ^[0-9]+$ ]] || die 'MIN_FREE_KIB must be a nonnegative integer'
  [[ "$GIT_TIMEOUT_SECONDS" =~ ^[0-9]+$ && "$GIT_TIMEOUT_SECONDS" -gt 0 ]] || die 'GIT_TIMEOUT_SECONDS must be positive'
  [[ "$CLONE_SINGLE_BRANCH" == true || "$CLONE_SINGLE_BRANCH" == false ]] || die 'Invalid CLONE_SINGLE_BRANCH'
  [[ "$CLONE_NO_TAGS" == true || "$CLONE_NO_TAGS" == false ]] || die 'Invalid CLONE_NO_TAGS'
  [[ "$CLONE_FILTER" == 'blob:none' || "$CLONE_FILTER" == 'none' ]] || die 'Invalid CLONE_FILTER'
  [[ "$GITHUB_BASE" == 'https://github.com' || "$GITHUB_BASE" == /* ]] || die 'GITHUB_BASE must be GitHub HTTPS (or offline fixture local path)'
  local available
  available="$(df -Pk "$REPOSITORY_ROOT" | awk 'END {print $4}')"
  [[ "$available" =~ ^[0-9]+$ && "$available" -ge "$MIN_FREE_KIB" ]] || die "Insufficient free disk: $available KiB"
  declare -A seen=()
  [[ "${#REPO_SPECS[@]}" -eq "$EXPECTED_JAVA_REPOS" ]] || die "Wrong URL mapping count: ${#REPO_SPECS[@]}"
  for row in "${REPO_SPECS[@]}"; do
    IFS='|' read -r project gh <<< "$row"
    [[ "$project" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]*$ && "$gh" =~ ^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$ ]] || die "Invalid GitHub mapping: $row"
    local label
    label="$(repo_label "$gh")"
    [[ ! -v "seen[$label]" ]] || die "Duplicate GitHub label: $label"
    seen[$label]=1
  done
}
count_projects() {
  local path="$1" count=0 row project gh
  for row in "${REPO_SPECS[@]}"; do
    IFS='|' read -r project gh <<< "$row"
    if [[ -d "$path/$(repo_label "$gh")" && ! -L "$path/$(repo_label "$gh")" ]]; then ((count+=1)); fi
  done
  echo "$count"
}
check_python() {
  local n
  [[ -d "$PYTHON_DIR" ]] || die "Missing Python destination: $PYTHON_DIR"
  n="$(find "$PYTHON_DIR" -mindepth 1 -maxdepth 1 -type d | wc -l)"
  [[ "$n" -eq "$EXPECTED_PYTHON_REPOS" ]] || die "Python count mismatch: expected $EXPECTED_PYTHON_REPOS, found $n"
}
write_manifest() {
  local row project gh url stage live archived d status sha tree shallow
  printf 'project\tgithub_url\tcurrent_location\thead_commit\thead_tree\tshallow\tsnapshot_archive\tstatus\n' > "$MANIFEST"
  for row in "${REPO_SPECS[@]}"; do
    IFS='|' read -r project gh <<< "$row"
    url="$GITHUB_BASE/$gh.git"
    live="$LIVE_DIR/$(repo_label "$gh")"
    stage="$STAGING_DIR/$(repo_label "$gh")"
    archived="$SNAPSHOT_ARCHIVE_DIR/$project"
    d=''; status='MISSING'; sha=''; tree=''; shallow=''
    if [[ -d "$live/.git" ]]; then d="$live"; status='ACTIVE';
    elif [[ -d "$stage/.git" ]]; then d="$stage"; status='STAGED';
    elif is_snapshot "$LIVE_DIR/$project"; then status='SNAPSHOT_ONLY'; fi
    if [[ -n "$d" ]]; then
      sha="$(git -C "$d" rev-parse HEAD)"
      tree="$(git -C "$d" rev-parse HEAD^{tree})"
      shallow="$(git -C "$d" rev-parse --is-shallow-repository)"
    fi
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$project" "$url" "$d" "$sha" "$tree" "$shallow" "$archived" "$status" >> "$MANIFEST"
  done
  sha256sum "$MANIFEST"
}
inspect_local() {
  local snap archived stage
  [[ -d "$LIVE_DIR" ]] || die "Missing Java snapshots: $LIVE_DIR"
  check_python
  snap="$(count_projects "$LIVE_DIR")"
  archived=0
  for row in "${REPO_SPECS[@]}"; do IFS='|' read -r project gh <<< "$row"; if is_snapshot "$SNAPSHOT_ARCHIVE_DIR/$project"; then ((archived+=1)); fi; done
  stage="$(count_projects "$STAGING_DIR")"
  local legacy=0 row project gh
  for row in "${REPO_SPECS[@]}"; do IFS='|' read -r project gh <<< "$row"; if is_snapshot "$LIVE_DIR/$project"; then ((legacy+=1)); fi; done
  [[ "$legacy" -eq "$EXPECTED_JAVA_REPOS" || "$archived" -eq "$EXPECTED_JAVA_REPOS" ]] || die "Original NiCad snapshots not found: legacy=$legacy archived=$archived"
  echo "timestamp_chicago=$(now)"
  echo "version=$VERSION mode=$MODE source=$LIVE_DIR"
  echo "java_live_git_project_count=$snap java_archived_snapshot_count=$archived staged_git_project_count=$stage"
  echo "legacy_java_snapshot_count=$(find "$LIVE_DIR" -mindepth 1 -maxdepth 1 -type d -name "*-java" | wc -l)"
  echo "python_repo_count=$EXPECTED_PYTHON_REPOS"
  echo "actual_github_clone_needed=$(( EXPECTED_JAVA_REPOS - stage ))"
  echo "clone_depth=$CLONE_DEPTH clone_filter=$CLONE_FILTER single_branch=$CLONE_SINGLE_BRANCH no_tags=$CLONE_NO_TAGS"
  echo 'NOTE: Upstream latest HEAD need not correspond to old NiCad source or old pair IDs.'
}
probe_all() {
  local row project gh url line sha failures=0
  printf 'project\tgithub_url\tremote_head\tstatus\n' > "$PROBE"
  for row in "${REPO_SPECS[@]}"; do
    IFS='|' read -r project gh <<< "$row"
    url="$GITHUB_BASE/$gh.git"
    echo "[$(now)] START step=probe project=$project url=$url"
    local out rc
    rc=0; out="$(timeout "$GIT_TIMEOUT_SECONDS" git ls-remote "$url" HEAD 2>&1)" || rc=$?
    if [[ "$rc" -eq 0 && "$out" =~ ^[0-9a-f]{40,64}[[:space:]]+HEAD ]]; then
      sha="${out%%[[:space:]]*}"
      printf '%s\t%s\t%s\tPASS\n' "$project" "$url" "$sha" >> "$PROBE"
      echo "[$(now)] PASS step=probe project=$project remote_head=$sha"
    else
      printf '%s\t%s\t\tFAIL\n' "$project" "$url" >> "$PROBE"
      echo "[$(now)] FAIL step=probe project=$project rc=$rc detail=${out:0:150}" >&2
      ((failures+=1))
    fi
  done
  echo "probe_failed_count=$failures probe_manifest=$PROBE"
  [[ "$failures" -eq 0 ]] || die "At least $failures GitHub URLs failed network probe; fix mapping before cloning"
}
clone_all() {
  local row project gh url dest temp status=0 n=0
  mkdir -p "$STAGING_DIR"
  for row in "${REPO_SPECS[@]}"; do
    IFS='|' read -r project gh <<< "$row"
    url="$GITHUB_BASE/$gh.git"
    dest="$STAGING_DIR/$(repo_label "$gh")"
    if [[ -d "$dest" ]]; then
      verify_repo "$dest" "$url" "$project"
      echo "[$(now)] PASS step=clone project=$project status=ALREADY_VERIFIED"
      ((n+=1)); continue
    fi
    [[ ! -e "$dest" && ! -L "$dest" ]] || die "Unexpected staged path: $dest"
    # A new unique incomplete clone directory is retained on failure for inspection.
    temp="$(mktemp -d "$STAGING_DIR/.partial-${project}-XXXXXX")"
    echo "[$(now)] START step=clone project=$project url=$url destination=$dest partial=$temp"
    local -a clone_args=()
    [[ "$CLONE_DEPTH" == '0' ]] || clone_args+=(--depth "$CLONE_DEPTH")
    [[ "$CLONE_SINGLE_BRANCH" != true ]] || clone_args+=(--single-branch)
    [[ "$CLONE_NO_TAGS" != true ]] || clone_args+=(--no-tags)
    [[ "$CLONE_FILTER" == 'none' ]] || clone_args+=(--filter="$CLONE_FILTER")
    timeout "$GIT_TIMEOUT_SECONDS" git clone "${clone_args[@]}" "$url" "$temp"
    verify_repo "$temp" "$url" "$project"
    mv -T -- "$temp" "$dest"
    echo "[$(now)] PASS step=clone project=$project sha=$(git -C "$dest" rev-parse HEAD)"
    ((n+=1))
  done
  echo "cloned_or_verified_total=$n"
}
verify_staged_all() {
  local row project gh url dest count=0
  for row in "${REPO_SPECS[@]}"; do
    IFS='|' read -r project gh <<< "$row"
    url="$GITHUB_BASE/$gh.git"
    dest="$STAGING_DIR/$(repo_label "$gh")"
    verify_repo "$dest" "$url" "$project"
    ((count+=1))
    echo "[$(now)] PASS step=verify_staged project=$project commit=$(git -C "$dest" rev-parse HEAD)"
  done
  [[ "$count" -eq "$EXPECTED_JAVA_REPOS" ]] || die "Stage count failed: $count"
}
preflight_activation() {
  local row project gh live staged saved url
  for row in "${REPO_SPECS[@]}"; do
    IFS='|' read -r project gh <<< "$row"
    live="$LIVE_DIR/$(repo_label "$gh")"; staged="$STAGING_DIR/$(repo_label "$gh")"; saved="$SNAPSHOT_ARCHIVE_DIR/$project"
    url="$GITHUB_BASE/$gh.git"
    if [[ -d "$live/.git" ]]; then
      verify_repo "$live" "$url" "$project"
      is_snapshot "$saved" || die "Missing archived snapshot of already active $project"
      [[ ! -e "$staged" ]] || die "Unexpected stage present for already active $project"
    else
      verify_repo "$staged" "$url" "$project"
      [[ ! -e "$live" && ! -L "$live" ]] || die "Java destination occupied: $live"
      if is_snapshot "$LIVE_DIR/$project"; then
        [[ ! -e "$saved" && ! -L "$saved" ]] || die "Snapshot archive conflict: $saved"
      elif [[ ! -e "$live" && ! -L "$live" && ! -e "$LIVE_DIR/$project" ]] && is_snapshot "$saved"; then
        : # Recovery from interruption after moving the old snapshot.
      else
        die "Cannot safely activate $project: live=$live saved=$saved staged=$staged"
      fi
    fi
  done
}
activate_all() {
  local row project gh live staged saved url
  mkdir -p "$SNAPSHOT_ARCHIVE_DIR"
  preflight_activation
  for row in "${REPO_SPECS[@]}"; do
    IFS='|' read -r project gh <<< "$row"
    live="$LIVE_DIR/$(repo_label "$gh")"; staged="$STAGING_DIR/$(repo_label "$gh")"; saved="$SNAPSHOT_ARCHIVE_DIR/$project"
    url="$GITHUB_BASE/$gh.git"
    if [[ -d "$live/.git" ]]; then
      echo "[$(now)] PASS step=activate project=$project status=ALREADY_ACTIVE"
      continue
    fi
    echo "[$(now)] START step=activate project=$project archive=$saved"
    if is_snapshot "$LIVE_DIR/$project"; then mv -T -- "$LIVE_DIR/$project" "$saved"; fi
    [[ ! -e "$live" && ! -L "$live" ]] || die "Refusing overwrite of existing Java live path: $live"
    if ! mv -T -- "$staged" "$live"; then
      if [[ ! -e "$LIVE_DIR/$project" && -d "$saved" ]]; then mv -T -- "$saved" "$LIVE_DIR/$project" || true; fi
      die "Failed to promote $project; recover from archive/stage before retrying"
    fi
    verify_repo "$live" "$url" "$project"
    echo "[$(now)] PASS step=activate project=$project sha=$(git -C "$live" rev-parse HEAD)"
  done
}
verify_active_all() {
  local row project gh url live saved count=0
  for row in "${REPO_SPECS[@]}"; do
    IFS='|' read -r project gh <<< "$row"
    url="$GITHUB_BASE/$gh.git"
    live="$LIVE_DIR/$(repo_label "$gh")"; saved="$SNAPSHOT_ARCHIVE_DIR/$project"
    verify_repo "$live" "$url" "$project"
    is_snapshot "$saved" || die "Missing original NiCad snapshot archive for $project"
    [[ ! -e "$STAGING_DIR/$(repo_label "$gh")" ]] || die "Unexpected staged copy for active $project"
    ((count+=1))
    echo "[$(now)] PASS step=verify_active project=$project sha=$(git -C "$live" rev-parse HEAD)"
  done
  [[ "$count" -eq "$EXPECTED_JAVA_REPOS" ]] || die "Wrong active count: $count"
}

echo "[$(now)] START experiment=$EXPERIMENT version=$VERSION mode=$MODE log=$LOG run=$RUN live=$LIVE_DIR staged=$STAGING_DIR archive=$SNAPSHOT_ARCHIVE_DIR"
start_step validate_inputs
check_specs
for cmd in git sha256sum find mktemp timeout tee; do command -v "$cmd" >/dev/null || die "Missing command: $cmd"; done
[[ -d "$REPOSITORY_ROOT" ]] || die "Missing repository root: $REPOSITORY_ROOT"
pass_step
case "$MODE" in
  --inspect)
    start_step inspect_repositories; inspect_local; write_manifest; pass_step ;;
  --probe)
    start_step probe_github_upstreams; probe_all; write_manifest; pass_step ;;
  --clone)
    start_step validate_local_layout; inspect_local; pass_step
    start_step clone_verified_github_repositories; clone_all; pass_step
    start_step verify_clones; verify_staged_all; write_manifest; pass_step ;;
  --verify)
    start_step verify_clones; verify_staged_all; write_manifest; pass_step ;;
  --activate)
    start_step promote_clones_preserving_nicad_snapshots; activate_all; write_manifest; pass_step ;;
  --verify-active)
    start_step verify_git_repositories_and_archives; verify_active_all; write_manifest; pass_step ;;
  --rename-python)
    start_step rename_python_git_repositories; rename_python; pass_step ;;
  --verify-python)
    start_step verify_python_git_repositories; python_inventory; pass_step ;;
esac
echo "[$(now)] PASS experiment=$EXPERIMENT version=$VERSION mode=$MODE status=VERIFIED_OR_COMPLETED"
