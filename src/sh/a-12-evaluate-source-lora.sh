#!/usr/bin/env bash
set -Eeuo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
mkdir -p "$ROOT/logs" "$ROOT/results/a-12"
STAMP="$(date +%Y%m%d-%H%M%S)-$$"
RUN_DIR="$ROOT/results/a-12/$STAMP"
mkdir -p "$RUN_DIR"
LOG="$ROOT/logs/a-12-evaluate-source-lora-$STAMP.log"
exec > >(tee "$LOG") 2>&1
trap 'rc=$?; echo "[A-12] exit=$rc log='$LOG' output='$RUN_DIR'"' EXIT
export TOKENIZERS_PARALLELISM=false
PY="$ROOT/src/py/a-12-evaluate-source-lora.py"
REVISION="${A12_REVISION:-bb9afde76d7945da5745592525db122d4d729eb1}"
MODEL="${A12_MODEL_ID:-bigcode/starcoder2-7b}"
ADAPTER="${A12_ADAPTER:-$ROOT/results/a-11/20261030-160737-397785/adapter}"
[[ -s "$ADAPTER/adapter_config.json" && -s "$ADAPTER/adapter_model.safetensors" ]] || { echo "[A-12] Missing adapter: $ADAPTER" >&2; exit 2; }
GPUS="${A12_GPUS:-0,1,2}"
IFS=',' read -r -a GPU_IDS <<< "$GPUS"
[[ ${#GPU_IDS[@]} -eq 3 ]] || { echo '[A-12] Specify three GPUs' >&2; exit 2; }
[[ "${GPU_IDS[0]}" != "${GPU_IDS[1]}" && "${GPU_IDS[0]}" != "${GPU_IDS[2]}" && "${GPU_IDS[1]}" != "${GPU_IDS[2]}" ]] || { echo '[A-12] GPU IDs must be distinct' >&2; exit 2; }
ARGS=(--output-dir "$RUN_DIR" --data-root "$ROOT/data/evaluation" --model-id "$MODEL" --revision "$REVISION" --adapter-dir "$ADAPTER" --max-input-tokens "${A12_MAX_INPUT_TOKENS:-12000}" --window-overlap "${A12_WINDOW_OVERLAP:-0.20}" --aggregation mean-margin)
echo "[A-12] Start $(date -Is); GPUs=$GPUS; adapter=$ADAPTER; run=$RUN_DIR"
python "$PY" preflight "${ARGS[@]}"
python "$PY" audit-lengths "${ARGS[@]}"
if [[ "${A12_PREFLIGHT_ONLY:-0}" == 1 ]]; then echo '[A-12] Preflight + token audit complete'; exit 0; fi
pids=()
for wi in 0 1 2; do
  echo "[A-12] worker $wi on GPU ${GPU_IDS[$wi]}"
  CUDA_VISIBLE_DEVICES="${GPU_IDS[$wi]}" python "$PY" infer "${ARGS[@]}" --worker-index "$wi" --num-workers 3 &
  pids+=("$!")
done
fail=0
for wi in 0 1 2; do
  if wait "${pids[$wi]}"; then echo "[A-12] worker $wi PASS"; else echo "[A-12] worker $wi FAIL"; fail=1; fi
done
[[ $fail == 0 ]] || exit 1
python "$PY" summarize "${ARGS[@]}" --num-workers 3
echo '[A-12] COMPLETE'
