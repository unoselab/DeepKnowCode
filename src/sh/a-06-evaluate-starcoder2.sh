#!/usr/bin/env bash
set -Eeuo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
mkdir -p "$ROOT/logs" "$ROOT/results/a-06"
STAMP="$(date +%Y%m%d-%H%M%S)-$$"
RUN_DIR="$ROOT/results/a-06/$STAMP"
mkdir -p "$RUN_DIR"
LOG="$ROOT/logs/a-06-evaluate-starcoder2-$STAMP.log"
exec > >(tee "$LOG") 2>&1
trap 'rc=$?; echo "[A-06] exit=$rc log='$LOG' output='$RUN_DIR'"' EXIT
export TOKENIZERS_PARALLELISM=false
PY="$ROOT/src/py/a-06-evaluate-starcoder2.py"
REVISION="${A06_REVISION:-bb9afde76d7945da5745592525db122d4d729eb1}"
GPUS="${A06_GPUS:-0,1,2}"
IFS=',' read -r -a GPU_IDS <<< "$GPUS"
[[ ${#GPU_IDS[@]} -eq 3 ]] || { echo "Specify three GPUs" >&2; exit 2; }
[[ "${GPU_IDS[0]}" != "${GPU_IDS[1]}" && "${GPU_IDS[0]}" != "${GPU_IDS[2]}" && "${GPU_IDS[1]}" != "${GPU_IDS[2]}" ]] || { echo 'GPU IDs must be distinct' >&2; exit 2; }
ARGS=(--output-dir "$RUN_DIR" --data-root "$ROOT/data/evaluation" --revision "$REVISION" --max-input-tokens "${A06_MAX_INPUT_TOKENS:-12000}")
echo "[A-06] Start $(date -Is), GPUs=$GPUS; run=$RUN_DIR"
python "$PY" preflight "${ARGS[@]}"
python "$PY" audit-lengths "${ARGS[@]}"
if [[ "${A06_PREFLIGHT_ONLY:-0}" == 1 ]]; then echo '[A-06] preflight + token audit complete'; exit 0; fi
pids=()
for wi in 0 1 2; do
  echo "[A-06] worker $wi on GPU ${GPU_IDS[$wi]}"
  CUDA_VISIBLE_DEVICES="${GPU_IDS[$wi]}" python "$PY" infer "${ARGS[@]}" --worker-index "$wi" --num-workers 3 --oversize error &
  pids+=("$!")
done
fail=0
for wi in 0 1 2; do
  if wait "${pids[$wi]}";then echo "worker $wi PASS";else echo "worker $wi FAIL";fail=1;fi
done
[[ $fail == 0 ]] || exit 1
python "$PY" summarize "${ARGS[@]}" --num-workers 3
echo '[A-06] Complete'
