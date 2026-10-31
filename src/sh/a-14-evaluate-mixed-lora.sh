#!/usr/bin/env bash
set -Eeuo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
mkdir -p "$ROOT/logs" "$ROOT/results/a-14"
STAMP="$(date +%Y%m%d-%H%M%S)-$$"
RUN_DIR="$ROOT/results/a-14/$STAMP"
mkdir -p "$RUN_DIR"
LOG="$ROOT/logs/a-14-evaluate-mixed-lora-$STAMP.log"
exec > >(tee "$LOG") 2>&1
trap 'rc=$?; echo "[A-14] exit=$rc log='$LOG' output='$RUN_DIR'"' EXIT
export TOKENIZERS_PARALLELISM=false
PY="$ROOT/src/py/a-14-evaluate-mixed-lora.py"
REVISION="${A14_REVISION:-bb9afde76d7945da5745592525db122d4d729eb1}"
MODEL="${A14_MODEL_ID:-bigcode/starcoder2-7b}"
ADAPTER="${A14_ADAPTER:-$ROOT/results/a-13/20261030-233707-413628/adapter}"
[[ -s "$ADAPTER/adapter_config.json" && -s "$ADAPTER/adapter_model.safetensors" ]] || { echo "[A-14] Missing adapter: $ADAPTER" >&2; exit 2; }
# Verify that evaluation uses the expected A-13 selected M2 adapter, not M1.
EXPECTED_ADAPTER_SHA="9dfbf49781d12dbcf8cb314a01d85328ec752c67eb1fc161010443671154cdcf"
ACTUAL_ADAPTER_SHA="$(sha256sum "$ADAPTER/adapter_model.safetensors" | awk '{print $1}')"
[[ "$ACTUAL_ADAPTER_SHA" == "$EXPECTED_ADAPTER_SHA" ]] || { echo "[A-14] Wrong adapter SHA-256: $ACTUAL_ADAPTER_SHA (expected $EXPECTED_ADAPTER_SHA)" >&2; exit 3; }
GPUS="${A14_GPUS:-0,1,2}"
IFS=',' read -r -a GPU_IDS <<< "$GPUS"
[[ ${#GPU_IDS[@]} -eq 3 ]] || { echo '[A-14] Specify three GPUs' >&2; exit 2; }
[[ "${GPU_IDS[0]}" != "${GPU_IDS[1]}" && "${GPU_IDS[0]}" != "${GPU_IDS[2]}" && "${GPU_IDS[1]}" != "${GPU_IDS[2]}" ]] || { echo '[A-14] GPU IDs must be distinct' >&2; exit 2; }
ARGS=(--output-dir "$RUN_DIR" --data-root "$ROOT/data/evaluation" --model-id "$MODEL" --revision "$REVISION" --adapter-dir "$ADAPTER" --max-input-tokens "${A14_MAX_INPUT_TOKENS:-12000}" --window-overlap "${A14_WINDOW_OVERLAP:-0.20}" --aggregation mean-margin)
echo "[A-14] Start $(date -Is); GPUs=$GPUS; adapter=$ADAPTER; run=$RUN_DIR"
python "$PY" preflight "${ARGS[@]}"
python "$PY" audit-lengths "${ARGS[@]}"
if [[ "${A14_PREFLIGHT_ONLY:-0}" == 1 ]]; then echo '[A-14] Preflight + token audit complete'; exit 0; fi
pids=()
for wi in 0 1 2; do
  echo "[A-14] worker $wi on GPU ${GPU_IDS[$wi]}"
  CUDA_VISIBLE_DEVICES="${GPU_IDS[$wi]}" python "$PY" infer "${ARGS[@]}" --worker-index "$wi" --num-workers 3 &
  pids+=("$!")
done
fail=0
for wi in 0 1 2; do
  if wait "${pids[$wi]}"; then echo "[A-14] worker $wi PASS"; else echo "[A-14] worker $wi FAIL"; fail=1; fi
done
[[ $fail == 0 ]] || exit 1
python "$PY" summarize "${ARGS[@]}" --num-workers 3
echo '[A-14] COMPLETE'
