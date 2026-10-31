#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
mkdir -p "$ROOT/logs"
STAMP="$(date -u +%Y%m%d-%H%M%S)-$$"
LOG="$ROOT/logs/b-01-discover-python-clones-$STAMP.log"
PYTHON_BIN="${PYTHON_BIN:-python}"
ENGINE="nicad"; ARGS=()
while (($#)); do
  case "$1" in
    --engine) ENGINE="${2:?missing engine}"; shift 2 ;;
    *) ARGS+=("$1"); shift ;;
  esac
done
if [[ "$ENGINE" == "ast" ]]; then
  "$PYTHON_BIN" "$ROOT/src/py/b-01-discover-python-clones.py" --root "$ROOT" "${ARGS[@]}" 2>&1 | tee "$LOG"
  exit "${PIPESTATUS[0]}"
fi


if [[ "$ENGINE" != "nicad" ]]; then
  echo "ERROR: This execution path requires --engine nicad" >&2
  exit 2
fi
{
  echo "B-01 | NiCad Python function clone discovery"

  NICAD_HOME="${NICAD_HOME:-$ROOT/tools/nicad-7.0.1-linux-x86_64}"
  TXL_HOME="${TXL_HOME:-$ROOT/tools/opentxl-11.3.7-linux-x64}"
  export PATH="$TXL_HOME/bin:$PATH"
  NICAD_BIN="${NICAD_BIN:-$NICAD_HOME/bin/nicad}"

  command -v txl >/dev/null || {
    echo "ERROR: OpenTxl not found" >&2
    exit 3
  }
  [[ -x "$NICAD_BIN" ]] || {
    echo "ERROR: NiCad executable not found: $NICAD_BIN" >&2
    exit 3
  }

  NICAD_BIN="$(realpath "$NICAD_BIN")"
  echo "NiCad: $NICAD_BIN"
  echo "OpenTxl: $(command -v txl)"
  txl -V

  PREP="$(
    "$PYTHON_BIN" "$ROOT/src/py/b-01-prepare-nicad.py" \
      --root "$ROOT" "${ARGS[@]}"
  )"

  printf '%s\n' "$PREP"

  OUTPUT="$(printf '%s\n' "$PREP" |
    sed -n 's/^OUTPUT_DIR=//p')"

  STAGE="$(printf '%s\n' "$PREP" |
    sed -n 's/^STAGE_DIR=//p')"

  [[ -n "$OUTPUT" && -d "$OUTPUT" &&
     -n "$STAGE" && -d "$STAGE" ]] || {
    echo "ERROR: Staging failed" >&2
    exit 4
  }

  echo "NiCad: functions py [staged source] default"

  (
    cd "$NICAD_HOME"
    "$NICAD_BIN" functions py "$STAGE" default
  )

  echo "[B-01] NiCad execution completed"
  echo "Experiment output: $OUTPUT"

} 2>&1 | tee "$LOG"
