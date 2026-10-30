#!/usr/bin/env bash
set -Eeuo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
mkdir -p "$ROOT/logs" "$ROOT/results/a-11"
STAMP="$(date +%Y%m%d-%H%M%S)-$$"
RUN="$ROOT/results/a-11/$STAMP"
mkdir "$RUN"
LOG="$ROOT/logs/a-11-train-source-lora-$STAMP.log"
exec > >(tee "$LOG") 2>&1
trap 'rc=$?; echo "[A-11] exit=$rc log='$LOG' output='$RUN'"' EXIT
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
GPUS="${A11_GPUS:-0,1,2}"
IFS=',' read -r -a GPU_IDS <<< "$GPUS"
[[ ${#GPU_IDS[@]} -eq 3 ]] || { echo '[A-11] Exactly 3 GPU IDs are required' >&2; exit 2; }
for id in "${GPU_IDS[@]}"; do
  [[ "$id" =~ ^[0-9]+$ ]] || { echo "[A-11] Invalid GPU ID: $id" >&2; exit 2; }
done
[[ "${GPU_IDS[0]}" != "${GPU_IDS[1]}" && "${GPU_IDS[0]}" != "${GPU_IDS[2]}" && "${GPU_IDS[1]}" != "${GPU_IDS[2]}" ]] || { echo '[A-11] GPU IDs must be distinct' >&2; exit 2; }
PY="$ROOT/src/py/a-11-train-source-lora.py"
[[ -f "$PY" ]] || { echo "[A-11] Missing $PY" >&2; exit 2; }
MAX_STEPS="${A11_MAX_STEPS:-500}"
EVAL_STEPS="${A11_EVAL_STEPS:-100}"
TRAIN_EXAMPLES="${A11_MAX_TRAIN_EXAMPLES:-12000}"
VAL_EXAMPLES="${A11_MAX_VAL_EXAMPLES:-512}"
if [[ "${A11_SMOKE:-0}" == 1 ]]; then
  MAX_STEPS=2; EVAL_STEPS=1; TRAIN_EXAMPLES=96; VAL_EXAMPLES=32
  echo '[A-11] SMOKE MODE: isolated 2-step DDP LoRA test'
fi
ARGS=(
  --root "$ROOT" --run-dir "$RUN"
  --revision "${A11_REVISION:-bb9afde76d7945da5745592525db122d4d729eb1}"
  --model-id "${A11_MODEL_ID:-bigcode/starcoder2-7b}"
  --max-seq-length "${A11_MAX_SEQ_LENGTH:-2048}"
  --max-train-examples "$TRAIN_EXAMPLES" --max-val-examples "$VAL_EXAMPLES"
  --sample-policy "${A11_SAMPLE_POLICY:-balanced}"
  --seed "${A11_SEED:-2026}"
  --max-steps "$MAX_STEPS" --eval-steps "$EVAL_STEPS"
  --log-steps "${A11_LOG_STEPS:-10}"
  --warmup-steps "${A11_WARMUP_STEPS:-20}"
  --per-device-batch-size "${A11_BATCH_SIZE:-1}"
  --per-device-eval-batch-size "${A11_EVAL_BATCH_SIZE:-1}"
  --gradient-accumulation-steps "${A11_GRAD_ACCUM:-8}"
  --learning-rate "${A11_LR:-0.0002}"
  --lora-rank "${A11_LORA_RANK:-16}"
  --lora-alpha "${A11_LORA_ALPHA:-32}"
  --lora-dropout "${A11_LORA_DROPOUT:-0.05}"
)
if [[ -n "${A11_A10_AUDIT:-}" ]]; then ARGS+=(--a10-audit "$A11_A10_AUDIT"); fi
echo "[A-11] Start $(date -Is); dataset=java_source; GPUs=$GPUS; run=$RUN"
echo '[A-11] Stage 1: audit guard, deterministic sample, full-prompt token-length filter'
python "$PY" prepare "${ARGS[@]}"
if [[ "${A11_PREPARE_ONLY:-0}" == 1 ]]; then
  echo '[A-11] PREPARE_ONLY complete. No model fine-tuning performed.'
  exit 0
fi
python - <<'PY'
try:
    import torch, transformers, peft, accelerate
except ImportError as exc:
    raise SystemExit(f'[A-11] Missing training dependency: {exc}. Install compatible peft/accelerate first.')
print('[A-11] Training dependencies loaded:', 'torch', torch.__version__, 'transformers', transformers.__version__, 'peft', peft.__version__, 'accelerate', accelerate.__version__, flush=True)
PY
export CUDA_VISIBLE_DEVICES="$GPUS"
echo '[A-11] Stage 2: DDP LoRA (three model replicas, synced gradients)'
python -m torch.distributed.run --standalone --nproc_per_node=3 "$PY" train "${ARGS[@]}"
echo '[A-11] Completed. No A-06 test labels have been accessed for training.'
