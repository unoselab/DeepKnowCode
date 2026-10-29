#!/usr/bin/env bash
set -Eeuo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
mkdir -p "$ROOT/logs" "$ROOT/results/a-04"
STAMP="$(date +%Y%m%d-%H%M%S)-$$"
LOG="$ROOT/logs/a-04-model-smoke-test-starcoder2-7b-${STAMP}.log"
exec > >(tee "$LOG") 2>&1
trap 'rc=$?; echo "[A-04] Exit code: $rc; Log: '$LOG'"' EXIT
export TOKENIZERS_PARALLELISM=false
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
# A04_GPUS: comma-separated physical GPU IDs; one independent model replica per GPU.
GPUS="${A04_GPUS:-0,1,2}"
IFS=',' read -r -a GPU_IDS <<< "$GPUS"
if [[ "${#GPU_IDS[@]}" -ne 3 ]]; then
  echo "[A-04] Expected exactly 3 GPU IDs in A04_GPUS, got: $GPUS" >&2; exit 2
fi
for i in "${GPU_IDS[@]}"; do
  if ! [[ "$i" =~ ^[0-9]+$ ]]; then echo "Invalid GPU ID: $i" >&2; exit 2; fi
done
if [[ "${GPU_IDS[0]}" == "${GPU_IDS[1]}" || "${GPU_IDS[0]}" == "${GPU_IDS[2]}" || "${GPU_IDS[1]}" == "${GPU_IDS[2]}" ]]; then
  echo "GPU IDs must be distinct" >&2; exit 2
fi
PY="$ROOT/src/py/a-04-model-smoke-test.py"
REVISION="${A04_REVISION:-bb9afde76d7945da5745592525db122d4d729eb1}"
OUT="$ROOT/results/a-04"
echo "[A-04] Start: $(date -Is); Model: StarCoder2-7B; GPUs: $GPUS; revision: $REVISION"
echo "[A-04] Phase 1: download/cache model only once"
python "$PY" --download-only --revision "$REVISION" "$@"
echo "[A-04] Phase 2: three simultaneous independent GPU replicas (data parallel)"
pids=()
for wi in 0 1 2; do
  gpu="${GPU_IDS[$wi]}"
  echo "[A-04] Launch worker $wi on physical GPU $gpu"
  CUDA_VISIBLE_DEVICES="$gpu" python "$PY" --model starcoder2-7b --revision "$REVISION" \
    --local-files-only --worker-index "$wi" --num-workers 3 --output-dir "$OUT" "$@" &
  pids+=("$!")
done
fail=0
for wi in 0 1 2; do
  if wait "${pids[$wi]}"; then
    echo "[A-04] Worker $wi PASS"
  else
    rc=$?
    echo "[A-04] Worker $wi FAIL exit=$rc"
    fail=1
  fi
done
if [[ "$fail" -ne 0 ]]; then exit 1; fi
echo "[A-04] All 3 workers complete. This is NON-SCORED diagnostic (not RQ3 F1)."
