#!/usr/bin/env bash
# B-05 v7 | NiCad Python grammar minimal reproduction (standalone wrapper; calls no other wrapper)
set -Eeuo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
mkdir -p "$ROOT/logs"
# Chicago local time. If tz data is unavailable, fall back to UTC and say so.
if TZ=America/Chicago date +%Y >/dev/null 2>&1 && [[ -e /usr/share/zoneinfo/America/Chicago ]]; then
  STAMP="$(TZ=America/Chicago date +%Y%m%d-%H%M%S)-$$"
else
  STAMP="$(date -u +%Y%m%d-%H%M%S)-$$"
  echo "WARNING: America/Chicago tz data not found; log name uses UTC" >&2
fi
LOG="$ROOT/logs/b-05-nicad-minimal-repro-$STAMP.log"

if [[ -z "${PYTHON_BIN:-}" ]]; then
  if command -v python >/dev/null 2>&1; then PYTHON_BIN=python; else PYTHON_BIN=python3; fi
fi

# Pass --help straight through (no lock, no log).
for arg in "$@"; do
  if [[ "$arg" == "-h" || "$arg" == "--help" || "$arg" == "--list-cases" ]]; then
    exec "$PYTHON_BIN" "$ROOT/src/py/b-05-nicad-minimal-repro.py" --root "$ROOT" "$@"
  fi
done

(
  trap 'code=$?; echo "[B-05] FAILED at shell line $LINENO; exit=$code" >&2' ERR
  echo "B-05 v7 | log=$LOG"

  command -v flock >/dev/null || { echo 'ERROR: flock required' >&2; exit 3; }
  exec 9>"$ROOT/logs/b-05-nicad.lock"
  flock -n 9 || { echo 'ERROR: another B-05 run is active' >&2; exit 6; }

  NICAD_HOME="${NICAD_HOME:-$ROOT/tools/nicad-7.0.1-linux-x86_64}"
  TXL_HOME="${TXL_HOME:-$ROOT/tools/opentxl-11.3.7-linux-x64}"
  export PATH="$TXL_HOME/bin:$PATH"
  [[ -x "$NICAD_HOME/bin/nicad" ]] || { echo "ERROR: NiCad not executable: $NICAD_HOME/bin/nicad" >&2; exit 3; }
  command -v txl >/dev/null || { echo "ERROR: txl missing; expected $TXL_HOME/bin/txl" >&2; exit 3; }
  NICAD_HOME="$(cd "$NICAD_HOME" && pwd)"
  TXL_HOME="$(cd "$TXL_HOME" && pwd)"
  echo "NiCad: $NICAD_HOME/bin/nicad"
  echo "TXL:   $(command -v txl)"
  # OpenTxl prints its version and may return status 1 for -V; informational only.
  TXL_VERSION_OUTPUT="$(txl -V 2>&1)" || TXL_VERSION_EXIT=$?
  printf '%s\n' "$TXL_VERSION_OUTPUT"
  [[ -z "${TXL_VERSION_EXIT:-}" ]] || echo "[B-05] NOTE: txl -V exit=${TXL_VERSION_EXIT} (informational only)"

  "$PYTHON_BIN" "$ROOT/src/py/b-05-nicad-minimal-repro.py" \
    --root "$ROOT" --nicad-home "$NICAD_HOME" --txl-home "$TXL_HOME" "$@"
) 2>&1 | tee "$LOG"
exit "${PIPESTATUS[0]}"
