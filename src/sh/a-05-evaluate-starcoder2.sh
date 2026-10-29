#!/usr/bin/env bash
# Run: 
# A05_PREFLIGHT_ONLY=1 bash src/sh/a-05-evaluate-starcoder2.sh
# bash src/sh/a-05-evaluate-starcoder2.sh

set -Eeuo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
mkdir -p "$ROOT/logs" "$ROOT/results/a-05"
STAMP="$(date +%Y%m%d-%H%M%S)-$$"
RUN_DIR="$ROOT/results/a-05/$STAMP"
mkdir -p "$RUN_DIR"
LOG="$ROOT/logs/a-05-evaluate-starcoder2-${STAMP}.log"
exec > >(tee "$LOG") 2>&1
trap 'rc=$?; echo "[A-05] exit=$rc log='$LOG' output='$RUN_DIR'"' EXIT
export TOKENIZERS_PARALLELISM=false
PY="$ROOT/src/py/a-05-evaluate-starcoder2.py"
REVISION="${A05_REVISION:-bb9afde76d7945da5745592525db122d4d729eb1}"
GPUS="${A05_GPUS:-0,1,2}"
IFS=',' read -r -a GPU_IDS <<< "$GPUS"
if [[ ${#GPU_IDS[@]} -ne 3 ]]; then echo "A05_GPUS requires exactly three GPU ids" >&2; exit 2; fi
if [[ ${GPU_IDS[0]} == "${GPU_IDS[1]}" || ${GPU_IDS[0]} == "${GPU_IDS[2]}" || ${GPU_IDS[1]} == "${GPU_IDS[2]}" ]]; then echo "GPU IDs must be distinct" >&2; exit 2; fi
for gpu in "${GPU_IDS[@]}"; do [[ "$gpu" =~ ^[0-9]+$ ]] || { echo "Invalid GPU id"; exit 2; }; done
ARGS=(--output-dir "$RUN_DIR" --revision "$REVISION")
echo "[A-05] Start $(date -Is), GPUs=$GPUS; run=$RUN_DIR"
python "$PY" preflight "${ARGS[@]}"
if [[ "${A05_PREFLIGHT_ONLY:-0}" == 1 ]]; then echo '[A-05] preflight-only requested'; exit 0; fi
pids=()
for wi in 0 1 2; do
  echo "[A-05] worker $wi -> GPU ${GPU_IDS[$wi]}"
  CUDA_VISIBLE_DEVICES="${GPU_IDS[$wi]}" python "$PY" infer "${ARGS[@]}" \
    --worker-index "$wi" --num-workers 3 \
    --max-input-tokens "${A05_MAX_INPUT_TOKENS:-4096}" \
    --oversize "${A05_OVERSIZE:-error}" &
  pids+=("$!")
done
fail=0
for wi in 0 1 2; do
  if wait "${pids[$wi]}"; then echo "[A-05] worker $wi PASS";
  else echo "[A-05] worker $wi FAIL";fail=1;fi
done
if [[ "$fail" != 0 ]]; then exit 1;fi
python "$PY" summarize "${ARGS[@]}" --num-workers 3
echo '[A-05] Complete'
