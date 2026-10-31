#!/usr/bin/env bash
set -Eeuo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
mkdir -p "$ROOT/logs"
STAMP="$(date -u +%Y%m%d-%H%M%S)-$$"
LOG="$ROOT/logs/b-01-discover-python-clones-$STAMP.log"
PYTHON_BIN="${PYTHON_BIN:-python}"
ENGINE=nicad
CONFIG="${NICAD_CONFIG:-default}"
ARGS=()
while (($#)); do
  case "$1" in
    --engine) ENGINE="${2:?missing engine}"; shift 2 ;;
    --config) CONFIG="${2:?missing config}"; shift 2 ;;
    *) ARGS+=("$1"); shift ;;
  esac
done
if [[ "$ENGINE" != nicad && "$ENGINE" != ast ]]; then echo 'ERROR: --engine must be nicad or ast' >&2; exit 2; fi
(
  trap 'code=$?; echo "[B-01] FAILED at shell line $LINENO; exit=$code" >&2' ERR
  echo "B-01 v7 | engine=$ENGINE | log=$LOG"
  if [[ "$ENGINE" == nicad ]]; then
    command -v flock >/dev/null || { echo 'ERROR: flock required for fixed staging/cache' >&2; exit 3; }
    exec 9>"$ROOT/logs/b-01-nicad.lock"
    flock -n 9 || { echo 'ERROR: another B-01 NiCad run is active' >&2; exit 6; }
  fi
  if [[ "$ENGINE" == ast ]]; then
    exec "$PYTHON_BIN" "$ROOT/src/py/b-01-discover-python-clones.py" --root "$ROOT" "${ARGS[@]}"
  fi
  NICAD_HOME="${NICAD_HOME:-$ROOT/tools/nicad-7.0.1-linux-x86_64}"
  TXL_HOME="${TXL_HOME:-$ROOT/tools/opentxl-11.3.7-linux-x64}"
  export PATH="$TXL_HOME/bin:$PATH"
  NICAD_BIN="${NICAD_BIN:-$NICAD_HOME/bin/nicad}"
  command -v txl >/dev/null || { echo "ERROR: txl missing; expected $TXL_HOME/bin/txl" >&2; exit 3; }
  [[ -x "$NICAD_BIN" ]] || { echo "ERROR: NiCad not executable: $NICAD_BIN" >&2; exit 3; }
  NICAD_BIN="$(realpath "$NICAD_BIN")"
  NICAD_HOME="$(cd "$(dirname "$NICAD_BIN")/.." && pwd)"
  echo "NiCad binary: $NICAD_BIN"
  echo "TXL binary: $(command -v txl)"
  # OpenTxl may print version successfully and return status 1 for -V.
  # Treat version output as informational; command presence and binary existence
  # are checked above, and NiCad itself determines runtime compatibility.
  TXL_VERSION_OUTPUT="$(txl -V 2>&1)" || TXL_VERSION_EXIT=$?
  printf '%s\n' "$TXL_VERSION_OUTPUT"
  if [[ -n "${TXL_VERSION_EXIT:-}" ]]; then
    echo "[B-01] NOTE: txl -V exit=${TXL_VERSION_EXIT} (informational only)"
  fi
  echo '[B-01] OpenTxl check completed.'
  echo '[B-01] Preflight: staging input and filtering test code...'
  PREP_OUTPUT="$(mktemp)"
  trap 'rm -f "${PREP_OUTPUT:-}"' EXIT
  "$PYTHON_BIN" "$ROOT/src/py/b-01-prepare-nicad.py" --root "$ROOT" "${ARGS[@]}" > "$PREP_OUTPUT"
  cat "$PREP_OUTPUT"
  OUTPUT="$(sed -n 's/^OUTPUT_DIR=//p' "$PREP_OUTPUT" | tail -n 1)"
  STAGE="$(sed -n 's/^STAGE_DIR=//p' "$PREP_OUTPUT" | tail -n 1)"
  [[ -n "$OUTPUT" && -d "$OUTPUT" && -n "$STAGE" && -d "$STAGE" ]] || { echo 'ERROR: staging missing OUTPUT_DIR or STAGE_DIR' >&2; exit 4; }
  # NiCad caches by the input directory basename. Clear only the B-01
  # cache for the fixed staged input (salt); never touch other cache roots.
  CACHE_BASE="$NICAD_HOME/nicadclones"
  CACHE_DIR="$CACHE_BASE/$(basename "$STAGE")"
  [[ "$(basename "$STAGE")" == "salt" ]] || {
    echo "ERROR: unexpected B-01 staging basename: $STAGE" >&2; exit 5;
  }
  [[ -d "$CACHE_BASE" && ! -L "$CACHE_BASE" ]] || {
    echo "ERROR: NiCad cache root missing/unsafe: $CACHE_BASE" >&2; exit 5;
  }
  [[ "$CACHE_DIR" == "$NICAD_HOME/nicadclones/salt" ]] || {
    echo "ERROR: refusing unsafe cache deletion: $CACHE_DIR" >&2; exit 5;
  }
  if [[ -L "$CACHE_DIR" ]]; then
    echo "ERROR: refusing to delete symlink cache: $CACHE_DIR" >&2; exit 5
  fi
  if [[ -d "$CACHE_DIR" ]]; then
    echo "[B-01] Removing previous NiCad cache (only): $CACHE_DIR"
    rm -rf -- "$CACHE_DIR"
  fi
  [[ ! -e "$CACHE_DIR" ]] || { echo "ERROR: cache still exists: $CACHE_DIR" >&2; exit 5; }
  echo "[B-01] Clean extraction cache: $CACHE_DIR"
  echo "[B-01] NiCad command: functions py $STAGE $CONFIG"
  # NiCad uses its own relative paths internally, so run from its installation root.
  # Output may be written next to the staged input rather than the working directory.
  (cd "$NICAD_HOME" && "$NICAD_BIN" functions py "$STAGE" "$CONFIG")
  echo '[B-01] Searching NiCad outputs...'
  "$PYTHON_BIN" "$ROOT/src/py/b-01-summarize-nicad.py" --output "$OUTPUT" --stage "$STAGE" --nicad-home "$NICAD_HOME" --config "$CONFIG"
  echo "[B-01 v7] COMPLETE. Results: $OUTPUT"
) 2>&1 | tee "$LOG"
exit "${PIPESTATUS[0]}"
