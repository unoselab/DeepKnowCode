#!/usr/bin/env bash
set -Eeuo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd)"
PY_SCRIPT="$ROOT/src/py2/d-01-controlled-python-lora.py"
if [[ -n "${D01_PYTHON:-}" ]]; then
  PYTHON="$D01_PYTHON"
elif [[ -x "$HOME/miniconda3/envs/deepknowcode/bin/python" ]]; then
  PYTHON="$HOME/miniconda3/envs/deepknowcode/bin/python"
else
  PYTHON="$(command -v python3)"
fi
[[ -f "$PY_SCRIPT" ]] || { echo "D-01 missing: $PY_SCRIPT" >&2; exit 2; }
MODE="${1:---dry-run}"
shift || true
case "$MODE" in
  --dry-run|--prepare|--train) ;;
  *) echo "Usage: bash src/sh2/d-01-controlled-python-lora.sh [--dry-run|--prepare|--train] [extra Python flags]" >&2; exit 2 ;;
esac
export TZ=America/Chicago
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
DATASET="${D01_DATASET:-data/training/clean/python_mixed}"
COMMON=(--root "$ROOT" --dataset "$DATASET" --model-id "${D01_MODEL_ID:-bigcode/starcoder2-7b}" \
    --revision "${D01_REVISION:-bb9afde76d7945da5745592525db122d4d729eb1}" \
    --max-seq-length "${D01_MAX_SEQ_LENGTH:-2048}" --max-steps "${D01_MAX_STEPS:-500}" \
    --expected-world-size "${D01_WORLD_SIZE:-3}" --batch-size "${D01_BATCH_SIZE:-1}" \
    --grad-accum "${D01_GRAD_ACCUM:-8}" --lr "${D01_LR:-0.0002}")
if [[ -n "${D01_APPROVAL:-}" ]]; then COMMON+=(--approval "$D01_APPROVAL"); fi
if [[ -n "${D01_INITIAL_ADAPTER:-}" ]]; then COMMON+=(--initial-adapter "$D01_INITIAL_ADAPTER"); fi
if [[ "${D01_ALLOW_DOWNLOAD:-0}" == 1 ]]; then COMMON+=(--allow-download); fi
if [[ "$MODE" == --dry-run ]]; then
  exec "$PYTHON" "$PY_SCRIPT" audit "${COMMON[@]}" "$@"
fi
mkdir -p "$ROOT/logs/d-01" "$ROOT/results/d-01"
if [[ "$MODE" == --prepare ]]; then
  STAMP="$(date +%Y%m%d-%H%M%S)-$$"
  RUN="$ROOT/results/d-01/$STAMP"
  mkdir -- "$RUN"
  LOG="$ROOT/logs/d-01/$STAMP.log"
  echo "D-01 RUN=$RUN"
  echo "D-01 LOG=$LOG"
  "$PYTHON" "$PY_SCRIPT" prepare "${COMMON[@]}" --run-dir "$RUN" "$@" 2>&1 | tee "$LOG"
  echo "D-01 PREPARED RUN=$RUN"
  exit 0
fi
[[ -n "${D01_RUN_DIR:-}" ]] || { echo 'D-01 --train requires D01_RUN_DIR pointing to the PREPARED run' >&2; exit 2; }
if [[ "$D01_RUN_DIR" = /* ]]; then RUN="$D01_RUN_DIR"; else RUN="$ROOT/$D01_RUN_DIR"; fi
[[ -f "$RUN/prepared/manifest.json" ]] || { echo "D-01 prepared manifest missing: $RUN" >&2; exit 2; }
[[ ! -e "$RUN/adapter" && ! -e "$RUN/training_summary.json" ]] || { echo 'D-01 training output already present; refusing overwrite' >&2; exit 2; }
GPUS="${D01_GPUS:-0,1,2}"
IFS=, read -r -a IDS <<< "$GPUS"
[[ ${#IDS[@]} -eq "${D01_WORLD_SIZE:-3}" ]] || { echo 'D-01 GPU count != world size' >&2; exit 2; }
for i in "${IDS[@]}"; do [[ "$i" =~ ^[0-9]+$ ]] || { echo 'D-01 invalid GPU ID' >&2; exit 2; }; done
[[ "$(printf '%s\n' "${IDS[@]}" | sort -u | wc -l)" -eq "${#IDS[@]}" ]] || { echo 'D-01 duplicate GPU IDs' >&2; exit 2; }
export CUDA_VISIBLE_DEVICES="$GPUS"
LOG="$ROOT/logs/d-01/$(basename "$RUN")-train.log"
echo "D-01 TRAIN RUN=$RUN LOG=$LOG GPUS=$GPUS"
"$PYTHON" -m torch.distributed.run --standalone --nproc_per_node="${D01_WORLD_SIZE:-3}" \
  "$PY_SCRIPT" train "${COMMON[@]}" --run-dir "$RUN" "$@" 2>&1 | tee "$LOG"
