#!/usr/bin/env bash
# E-01b v1 independent shell entrypoint; does NOT call an existing shell wrapper.

# bash src/sh2/e-01b-train-source-lora.sh --preflight

# E01B_GPUS=0,1,2 \
# bash src/sh2/e-01b-train-source-lora.sh --smoke

# E01B_GPUS=0,1,2 \
# bash src/sh2/e-01b-train-source-lora.sh --train

set -Eeuo pipefail
export TZ=America/Chicago
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PY="$ROOT/src/py2/e-01b-train-source-lora.py"
[[ -f "$PY" ]] || { echo "E-01b missing $PY" >&2; exit 2; }
MODE="${1:---preflight}"
if (( $# )); then shift; fi
case "$MODE" in
  --preflight) SUB=preflight ;;
  --prepare) SUB=prepare ;;
  --smoke) SUB=smoke ;;
  --train) SUB=train ;;
  *) echo 'Usage: e-01b-train-source-lora.sh --preflight|--prepare|--smoke|--train' >&2; exit 2 ;;
esac
[[ $# -eq 0 ]] || { echo 'Unexpected positional arguments' >&2; exit 2; }
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
GPUS="${E01B_GPUS:-0,1,2}"
IFS=',' read -r -a GPU_IDS <<< "$GPUS"
[[ ${#GPU_IDS[@]} -eq 3 ]] || { echo 'Require three GPU IDs' >&2; exit 2; }
for id in "${GPU_IDS[@]}"; do [[ "$id" =~ ^[0-9]+$ ]] || { echo 'Invalid GPU ID' >&2; exit 2; }; done
[[ "${GPU_IDS[0]}" != "${GPU_IDS[1]}" && "${GPU_IDS[0]}" != "${GPU_IDS[2]}" && "${GPU_IDS[1]}" != "${GPU_IDS[2]}" ]] || { echo 'Duplicate GPU ID' >&2; exit 2; }
export CUDA_VISIBLE_DEVICES="$GPUS"
if [[ "$SUB" == 'preflight' ]]; then
  exec python "$PY" preflight --root "$ROOT"
fi
mkdir -p "$ROOT/results/e-01b" "$ROOT/logs/e-01b"
STAMP="$(date +%Y%m%d-%H%M%S)-$$"
RUN="$ROOT/results/e-01b/$STAMP-$SUB"
LOG="$ROOT/logs/e-01b/$STAMP-$SUB.log"
[[ ! -e "$RUN" ]] || { echo "Run exists $RUN" >&2; exit 2; }
exec > >(tee "$LOG") 2>&1
trap 'rc=$?; echo "E-01b exit=$rc RUN='$RUN' LOG='$LOG'"' EXIT
python "$PY" "$SUB" --root "$ROOT" --run-dir "$RUN"
