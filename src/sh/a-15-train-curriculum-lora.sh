#!/usr/bin/env bash
set -Eeuo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
mkdir -p "$ROOT/logs" "$ROOT/results/a-15"
STAMP="$(date +%Y%m%d-%H%M%S)-$$"
RUN="$ROOT/results/a-15/$STAMP"
mkdir "$RUN"
LOG="$ROOT/logs/a-15-train-mixed-lora-$STAMP.log"
exec > >(tee "$LOG") 2>&1
trap 'rc=$?; echo "[A-15] exit=$rc log='$LOG' output='$RUN'"' EXIT
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
GPUS="${A15_GPUS:-0,1,2}"
IFS=',' read -r -a GPU_IDS <<< "$GPUS"
[[ ${#GPU_IDS[@]} -eq 3 ]] || { echo '[A-15] Exactly 3 GPU IDs are required' >&2; exit 2; }
for id in "${GPU_IDS[@]}"; do
  [[ "$id" =~ ^[0-9]+$ ]] || { echo "[A-15] Invalid GPU ID: $id" >&2; exit 2; }
done
[[ "${GPU_IDS[0]}" != "${GPU_IDS[1]}" && "${GPU_IDS[0]}" != "${GPU_IDS[2]}" && "${GPU_IDS[1]}" != "${GPU_IDS[2]}" ]] || { echo '[A-15] GPU IDs must be distinct' >&2; exit 2; }
PY="$ROOT/src/py/a-15-train-curriculum-lora.py"
[[ -f "$PY" ]] || { echo "[A-15] Missing $PY" >&2; exit 2; }
MAX_STEPS="${A15_MAX_STEPS:-500}"
EVAL_STEPS="${A15_EVAL_STEPS:-100}"
SOURCE_TRAIN="${A15_SOURCE_TRAIN:-11000}"
JAVA_TRAIN="${A15_JAVA_TARGET_TRAIN:-800}"
PYTHON_TRAIN="${A15_PYTHON_TARGET_TRAIN:-200}"
SOURCE_VAL="${A15_SOURCE_VAL:-412}"
JAVA_VAL="${A15_JAVA_TARGET_VAL:-60}"
PYTHON_VAL="${A15_PYTHON_TARGET_VAL:-40}"
INITIAL_ADAPTER="${A15_INITIAL_ADAPTER:-$ROOT/results/a-11/20261030-160737-397785/adapter}"
if [[ "${A15_SMOKE:-0}" == 1 ]]; then
  echo '[A-15] SMOKE is not supported for the fixed comparison schedule. Use PREPARE_ONLY.' >&2
  exit 2
fi
ARGS=(
  --root "$ROOT" --run-dir "$RUN"
  --revision "${A15_REVISION:-bb9afde76d7945da5745592525db122d4d729eb1}"
  --model-id "${A15_MODEL_ID:-bigcode/starcoder2-7b}"
  --max-seq-length "${A15_MAX_SEQ_LENGTH:-2048}"
  --source-train "$SOURCE_TRAIN" --java-target-train "$JAVA_TRAIN" --python-target-train "$PYTHON_TRAIN"
  --source-val "$SOURCE_VAL" --java-target-val "$JAVA_VAL" --python-target-val "$PYTHON_VAL"
  --initial-adapter "$INITIAL_ADAPTER"
  --reference-prepared "${A15_REFERENCE_PREPARED:-$ROOT/results/a-13/20261030-233707-413628/prepared}"
  --sample-policy "${A15_SAMPLE_POLICY:-balanced}"
  --seed "${A15_SEED:-2026}"
  --max-steps "$MAX_STEPS" --eval-steps "$EVAL_STEPS"
  --log-steps "${A15_LOG_STEPS:-10}"
  --warmup-steps "${A15_WARMUP_STEPS:-20}"
  --per-device-batch-size "${A15_BATCH_SIZE:-1}"
  --per-device-eval-batch-size "${A15_EVAL_BATCH_SIZE:-1}"
  --gradient-accumulation-steps "${A15_GRAD_ACCUM:-8}"
  --learning-rate "${A15_LR:-0.0002}"
  --lora-rank "${A15_LORA_RANK:-16}"
  --lora-alpha "${A15_LORA_ALPHA:-32}"
  --lora-dropout "${A15_LORA_DROPOUT:-0.05}"
)

if [[ -n "${A15_A10_AUDIT:-}" ]]; then ARGS+=(--a10-audit "$A15_A10_AUDIT"); fi
echo "[A-15] Start $(date -Is); dataset=A13-identical-pairs+curriculum-proxy; GPUs=$GPUS; run=$RUN"
echo "[A-15] Requested fixed total mix source=$SOURCE_TRAIN java_target=$JAVA_TRAIN python_target=$PYTHON_TRAIN"
echo '[A-15] Stage 1: audit guard, deterministic sample, full-prompt token-length filter'
python "$PY" prepare "${ARGS[@]}"
if [[ "${A15_PREPARE_ONLY:-0}" == 1 ]]; then
  echo '[A-15] PREPARE_ONLY complete. No model fine-tuning performed.'
  exit 0
fi
python - <<'PY'
try:
    import torch, transformers, peft, accelerate
except ImportError as exc:
    raise SystemExit(f'[A-15] Missing training dependency: {exc}. Install compatible peft/accelerate first.')
print('[A-15] Training dependencies loaded:', 'torch', torch.__version__, 'transformers', transformers.__version__, 'peft', peft.__version__, 'accelerate', accelerate.__version__, flush=True)
PY
export CUDA_VISIBLE_DEVICES="$GPUS"
echo '[A-15] Stage 2: DDP LoRA with sequential phase ordering (three replicas, synced gradients)'
python -m torch.distributed.run --standalone --nproc_per_node=3 "$PY" train "${ARGS[@]}"
echo '[A-15] Completed. No A-06 test labels have been accessed for training.'
