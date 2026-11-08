#!/usr/bin/env bash
# Standalone D-04 v7 M3 Trainer order audit and BF16 LoRA training.
# DO NOT start GPU training until --audit-trainer passes.
# Step 1 (CPU Trainer/Accelerate audit; 3 ranks, 3 optimizer steps):
# bash src/sh2/d-04-train-curriculum-lora.sh --audit-trainer
# Step 2 (inspect results/d-04/trainer-order-audit-v7/manifest.json).
# Step 3 (NEW GPU training; independent M1 adapter, never M2):
# bash src/sh2/d-04-train-curriculum-lora.sh --train
set -Eeuo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYFILE="$ROOT/src/py2/d-04-train-curriculum-lora.py"
[[ -f "$PYFILE" ]] || { echo "Missing M3 training script: $PYFILE" >&2; exit 2; }
MODE="${1:---audit-trainer}"
shift || true
AUDIT_STEPS=3
GPU_IDS="${D04_GPUS:-0,1,2}"
RUN=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --audit-steps) AUDIT_STEPS="$2";shift 2;;
    --gpus) GPU_IDS="$2";shift 2;;
    --run-dir) RUN="$2";shift 2;;
    *) echo "Unknown option: $1" >&2;exit 2;;
  esac
done
[[ "$MODE" == --audit-trainer || "$MODE" == --train ]] || { echo 'Usage: bash src/sh2/d-04-train-curriculum-lora.sh --audit-trainer [--audit-steps N] | --train [--run-dir PATH] [--gpus 0,1,2]' >&2; exit 2; }
[[ "$AUDIT_STEPS" =~ ^[0-9]+$ ]] && (( AUDIT_STEPS>=1 && AUDIT_STEPS<=811 )) || { echo 'audit-steps must be 1..811' >&2;exit 2; }
export TZ=America/Chicago
STAMP="$(date +%Y%m%d-%H%M%S)-$$"
mkdir -p "$ROOT/logs/d-04"
LOG="$ROOT/logs/d-04/d-04-$STAMP.log"
exec > >(tee -a "$LOG") 2>&1
trap 'ret=$?; echo "[D-04 v7] exit=$ret mode=$MODE run=$RUN log=$LOG chicago=$(date -Is)"' EXIT
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
if [[ "$MODE" == --audit-trainer ]];then
  [[ -z "$RUN" ]] || { echo '--run-dir cannot override fixed audit evidence directory' >&2;exit 2; }
  RUN="$ROOT/results/d-04/trainer-order-audit-v7"
  [[ ! -e "$RUN" ]] || { echo "Audit path exists, refusing to overwrite: $RUN" >&2;exit 2; }
  mkdir -p "$RUN"
  echo "[D-04 v7] CPU Trainer/Accelerate DataLoader order audit: $(date -Is)"
  python -m torch.distributed.run --standalone --nproc_per_node=3 "$PYFILE" audit-trainer \
    --root "$ROOT" --run-dir "$RUN" --audit-steps "$AUDIT_STEPS"
  exit 0
fi
[[ -f "$ROOT/results/d-04/trainer-order-audit-v7/manifest.json" ]] || { echo 'Missing CPU Trainer audit evidence. Run --audit-trainer first.' >&2;exit 2; }
IFS=, read -r -a ids <<< "$GPU_IDS"
[[ ${#ids[@]} -eq 3 ]] || { echo 'Exactly 3 GPUs required' >&2;exit 2; }
for id in "${ids[@]}";do [[ "$id" =~ ^[0-9]+$ ]] || exit 2;done
[[ "${ids[0]}" != "${ids[1]}" && "${ids[0]}" != "${ids[2]}" && "${ids[1]}" != "${ids[2]}" ]] || exit 2
if [[ -z "$RUN" ]];then RUN="$ROOT/results/d-04/m3-curriculum-multilingual-$STAMP";fi
[[ ! -e "$RUN" ]] || { echo "Refusing to overwrite previous M3 run: $RUN" >&2;exit 2; }
mkdir -p "$RUN"
export CUDA_VISIBLE_DEVICES="$GPU_IDS"
python - <<'PY'
import torch,transformers,peft,accelerate
print('[D-04 v7] dependencies:',torch.__version__,transformers.__version__,peft.__version__,accelerate.__version__)
PY
echo "[D-04 v7] starting M3: prepared=balanced_multilingual candidate=curriculum-audit-v5 initial=M1 order=$RUN"
python -m torch.distributed.run --standalone --nproc_per_node=3 "$PYFILE" train \
  --root "$ROOT" --run-dir "$RUN"
