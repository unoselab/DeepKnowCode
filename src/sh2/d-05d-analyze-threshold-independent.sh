#!/usr/bin/env bash
# D-05d v1 standalone runner: does not invoke earlier shell wrappers.

# bash src/sh2/d-05d-analyze-threshold-independent.sh \
#   --dry-run \
#   --input results/d-05c/20261109-004326-58663

# bash src/sh2/d-05d-analyze-threshold-independent.sh \
#   --analyze \
#   --input results/d-05c/20261109-004326-58663

set -Eeuo pipefail
export TZ=America/Chicago
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PY="$ROOT/src/py2/d-05d-analyze-threshold-independent.py"
[[ -f "$PY" ]] || { echo "Missing D-05d Python script: $PY" >&2; exit 2; }
MODE="${1:---dry-run}"
[[ "$MODE" == --dry-run || "$MODE" == --analyze ]] || { echo "Usage: $0 --dry-run|--analyze --input D05C_RUN_DIRECTORY_OR_ZIP [--out NEW_DIRECTORY]" >&2; exit 2; }
shift || true
ARGS=(--root "$ROOT")
HAS_INPUT=0
while (( $# )); do
    case "$1" in
        --input|--out)
            [[ $# -ge 2 ]] || { echo "Missing argument for $1" >&2; exit 2; }
            [[ "$1" != --input ]] || HAS_INPUT=1
            ARGS+=("$1" "$2"); shift 2;;
        *) echo "Unknown argument: $1" >&2; exit 2;;
    esac
done
[[ $HAS_INPUT == 1 ]] || { echo "Missing --input. Pass exact D-05c run directory or ZIP." >&2; exit 2; }
if [[ "$MODE" == --dry-run ]]; then
    exec python "$PY" --dry-run "${ARGS[@]}"
fi
exec python "$PY" "${ARGS[@]}"
