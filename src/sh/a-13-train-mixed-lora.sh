#!/usr/bin/env bash
set -Eeuo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
mkdir -p "$ROOT/logs" "$ROOT/results/a-13"
STAMP="$(date +%Y%m%d-%H%M%S)-$$"
RUN="$ROOT/results/a-13/$STAMP"
mkdir "$RUN"
LOG="$ROOT/logs/a-13-train-mixed-lora-$STAMP.log"
exec > >(tee "$LOG") 2>&1
trap 'rc=$?; echo "[A-13] exit=$rc log='$LOG' output='$RUN'"' EXIT
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
GPUS="${A13_GPUS:-0,1,2}"
IFS=',' read -r -a GPU_IDS <<< "$GPUS"
[[ ${#GPU_IDS[@]} -eq 3 ]] || { echo '[A-13] Exactly 3 GPU IDs are required' >&2; exit 2; }
for id in "${GPU_IDS[@]}"; do
  [[ "$id" =~ ^[0-9]+$ ]] || { echo "[A-13] Invalid GPU ID: $id" >&2; exit 2; }
done
[[ "${GPU_IDS[0]}" != "${GPU_IDS[1]}" && "${GPU_IDS[0]}" != "${GPU_IDS[2]}" && "${GPU_IDS[1]}" != "${GPU_IDS[2]}" ]] || { echo '[A-13] GPU IDs must be distinct' >&2; exit 2; }
PY="$ROOT/src/py/a-13-train-mixed-lora.py"
[[ -f "$PY" ]] || { echo "[A-13] Missing $PY" >&2; exit 2; }
MAX_STEPS="${A13_MAX_STEPS:-500}"
EVAL_STEPS="${A13_EVAL_STEPS:-100}"
SOURCE_TRAIN="${A13_SOURCE_TRAIN:-11000}"
JAVA_TRAIN="${A13_JAVA_TARGET_TRAIN:-800}"
PYTHON_TRAIN="${A13_PYTHON_TARGET_TRAIN:-200}"
SOURCE_VAL="${A13_SOURCE_VAL:-412}"
JAVA_VAL="${A13_JAVA_TARGET_VAL:-60}"
PYTHON_VAL="${A13_PYTHON_TARGET_VAL:-40}"
INITIAL_ADAPTER="${A13_INITIAL_ADAPTER:-$ROOT/results/a-11/20261030-160737-397785/adapter}"
if [[ "${A13_SMOKE:-0}" == 1 ]]; then
  MAX_STEPS=2; EVAL_STEPS=1; SOURCE_TRAIN=48; JAVA_TRAIN=24; PYTHON_TRAIN=24; SOURCE_VAL=16; JAVA_VAL=8; PYTHON_VAL=8
  echo '[A-13] SMOKE MODE: isolated 2-step DDP LoRA test'
fi
ARGS=(
  --root "$ROOT" --run-dir "$RUN"
  --revision "${A13_REVISION:-bb9afde76d7945da5745592525db122d4d729eb1}"
  --model-id "${A13_MODEL_ID:-bigcode/starcoder2-7b}"
  --max-seq-length "${A13_MAX_SEQ_LENGTH:-2048}"
  --source-train "$SOURCE_TRAIN" --java-target-train "$JAVA_TRAIN" --python-target-train "$PYTHON_TRAIN"
  --source-val "$SOURCE_VAL" --java-target-val "$JAVA_VAL" --python-target-val "$PYTHON_VAL"
  --initial-adapter "$INITIAL_ADAPTER"
  --sample-policy "${A13_SAMPLE_POLICY:-balanced}"
  --seed "${A13_SEED:-2026}"
  --max-steps "$MAX_STEPS" --eval-steps "$EVAL_STEPS"
  --log-steps "${A13_LOG_STEPS:-10}"
  --warmup-steps "${A13_WARMUP_STEPS:-20}"
  --per-device-batch-size "${A13_BATCH_SIZE:-1}"
  --per-device-eval-batch-size "${A13_EVAL_BATCH_SIZE:-1}"
  --gradient-accumulation-steps "${A13_GRAD_ACCUM:-8}"
  --learning-rate "${A13_LR:-0.0002}"
  --lora-rank "${A13_LORA_RANK:-16}"
  --lora-alpha "${A13_LORA_ALPHA:-32}"
  --lora-dropout "${A13_LORA_DROPOUT:-0.05}"
)
if [[ "${A13_SMOKE:-0}" == 1 || "${A13_ALLOW_CUSTOM_BUDGET:-0}" == 1 ]]; then ARGS+=(--allow-custom-budget); fi
if [[ -n "${A13_A10_AUDIT:-}" ]]; then ARGS+=(--a10-audit "$A13_A10_AUDIT"); fi
echo "[A-13] Start $(date -Is); dataset=source+java_target+python_target; GPUs=$GPUS; run=$RUN"
echo "[A-13] Requested static mix source=$SOURCE_TRAIN java_target=$JAVA_TRAIN python_target=$PYTHON_TRAIN"
echo '[A-13] Stage 1: audit guard, deterministic sample, full-prompt token-length filter'
python "$PY" prepare "${ARGS[@]}"
if [[ "${A13_PREPARE_ONLY:-0}" == 1 ]]; then
  echo '[A-13] PREPARE_ONLY complete. No model fine-tuning performed.'
  exit 0
fi
python - <<'PY'
try:
    import torch, transformers, peft, accelerate
except ImportError as exc:
    raise SystemExit(f'[A-13] Missing training dependency: {exc}. Install compatible peft/accelerate first.')
print('[A-13] Training dependencies loaded:', 'torch', torch.__version__, 'transformers', transformers.__version__, 'peft', peft.__version__, 'accelerate', accelerate.__version__, flush=True)
PY
export CUDA_VISIBLE_DEVICES="$GPUS"
echo '[A-13] Stage 2: DDP LoRA (three model replicas, synced gradients)'
python -m torch.distributed.run --standalone --nproc_per_node=3 "$PY" train "${ARGS[@]}"
echo '[A-13] Completed. No A-06 test labels have been accessed for training.'
