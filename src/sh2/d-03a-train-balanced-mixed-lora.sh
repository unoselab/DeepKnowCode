#!/usr/bin/env bash
# Standalone D-03 v5 M2 launcher. Never invokes any other shell wrapper.
set -Eeuo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYFILE="$ROOT/src/py2/d-03-train-balanced-mixed-lora.py"
[[ -f "$PYFILE" ]] || { echo "Missing Python script: $PYFILE" >&2; exit 2; }
MODE="${1:---audit}"
shift || true
TARGET=3000
POLICY=fixed-source
ADAPTER=""
RUN=""
GPUS="${D03_GPUS:-0,1,2}"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --target) TARGET="$2";shift 2;;
    --budget-policy) POLICY="$2";shift 2;;
    --adapter) ADAPTER="$2";shift 2;;
    --run-dir) RUN="$2";shift 2;;
    --gpus) GPUS="$2";shift 2;;
    *) echo "Unknown option: $1" >&2;exit 2;;
  esac
done
case "$MODE" in --audit|--audit-max|--prepare|--train|--resume-train) ;; *) echo "Usage: bash src/sh2/d-03-train-balanced-mixed-lora.sh --audit|--audit-max|--prepare|--train|--resume-train [--target EVEN_INTEGER] [--budget-policy fixed-total|fixed-source] [--adapter PATH] [--run-dir PATH]" >&2;exit 2;;esac
[[ "$TARGET" =~ ^[0-9]+$ ]] && (( TARGET > 0 && TARGET % 2 == 0 )) || { echo "Target must be a positive even integer" >&2;exit 2; }
[[ "$POLICY" == fixed-total || "$POLICY" == fixed-source ]] || { echo "Invalid budget policy" >&2;exit 2; }
export TZ=America/Chicago
STAMP="$(date +%Y%m%d-%H%M%S)-$$"
if [[ -z "$RUN" ]];then RUN="$ROOT/results/d-03/$STAMP";fi
if [[ "$MODE" == --resume-train ]];then
  [[ -f "$RUN/prepared/manifest.json" ]] || { echo "Existing --run-dir with prepared manifest required" >&2;exit 2; }
else
  [[ ! -e "$RUN" ]] || { echo "Refusing to reuse existing run directory: $RUN" >&2;exit 2; }
  mkdir -p "$RUN"
fi
mkdir -p "$ROOT/logs"
LOG="$ROOT/logs/d-03-$STAMP.log"
exec > >(tee -a "$LOG") 2>&1
trap 'ret=$?; echo "[D-03] exit=$ret run=$RUN log=$LOG chicago=$(date -Is)"' EXIT
printf '[D-03 v5] start=%s mode=%s target_per_language=%s policy=%s run=%s\n' "$(date -Is)" "$MODE" "$TARGET" "$POLICY" "$RUN"
COMMON=(--root "$ROOT" --run-dir "$RUN" --target-per-language "$TARGET" --budget-policy "$POLICY")
if [[ "$MODE" == --audit || "$MODE" == --audit-max ]];then
  CMD=audit
  [[ "$MODE" == --audit-max ]] && CMD=audit-max
  python "$PYFILE" "$CMD" "${COMMON[@]}"
  exit 0
fi
if [[ "$MODE" != --resume-train ]];then
  if [[ -z "$ADAPTER" ]];then
    # Explicitly pinned M1 run is first choice; fall back to the newest valid A-11 adapter.
    CANDIDATE="$ROOT/results/a-11/20261030-160737-397785/adapter"
    if [[ -f "$CANDIDATE/adapter_model.safetensors" && -f "$CANDIDATE/adapter_config.json" ]];then
      ADAPTER="$CANDIDATE"
    else
      while IFS= read -r path;do
        if [[ -f "$path/adapter_config.json" && -f "$path/adapter_model.safetensors" ]];then ADAPTER="$path";break;fi
      done < <(find "$ROOT/results/a-11" -mindepth 2 -maxdepth 2 -type d -name adapter 2>/dev/null | sort -r)
    fi
  fi
  [[ -n "$ADAPTER" ]] || { echo 'M1 adapter not found. Supply --adapter PATH.' >&2;exit 2; }
  echo "[D-03] M1 initial adapter: $ADAPTER"
  python "$PYFILE" prepare "${COMMON[@]}" --initial-adapter "$ADAPTER"
  if [[ "$MODE" == --prepare ]];then echo "[D-03] PREPARED only; training not started";exit 0;fi
fi
IFS=, read -r -a ids <<< "$GPUS"
[[ "${#ids[@]}" -eq 3 ]] || { echo 'Exactly 3 GPU IDs required' >&2;exit 2; }
for id in "${ids[@]}";do [[ "$id" =~ ^[0-9]+$ ]] || exit 2;done
[[ "${ids[0]}" != "${ids[1]}" && "${ids[0]}" != "${ids[2]}" && "${ids[1]}" != "${ids[2]}" ]] || exit 2
export CUDA_VISIBLE_DEVICES="$GPUS"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
python - <<'PY'
import torch,transformers,peft,accelerate
print('[D-03] dependency versions:',torch.__version__,transformers.__version__,peft.__version__,accelerate.__version__)
PY
python -m torch.distributed.run --standalone --nproc_per_node=3 "$PYFILE" train "${COMMON[@]}"
