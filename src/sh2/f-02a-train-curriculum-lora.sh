#!/usr/bin/env bash
# F-02a v3: standalone 3-GPU NCCL Curriculum audit, DataLoader smoke, and real 2-step LoRA model smoke.
# All experiment configuration lives here. Immutable F-01c and F-01e pins required.
# Usage: bash src/sh2/f-02a-train-curriculum-lora.sh --audit
#        bash src/sh2/f-02a-train-curriculum-lora.sh --smoke
# --train performs standalone full 811-step M3 curriculum training after hard preflight gate.
set -Eeuo pipefail
export TZ=America/Chicago
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
PYTHON_BIN="python"
PYFILE="$ROOT/src/py2/f-02a-curriculum-ddp.py"
MODE="${1:---audit}"
if (($#)); then shift; fi
[[ $# -eq 0 ]] || { echo 'Unexpected arguments' >&2; exit 2; }
case "$MODE" in
  --audit|--smoke|--train-smoke|--train) ;;
  *) echo 'Usage: --audit | --smoke | --train-smoke | --train' >&2; exit 2 ;;
esac

# ----- Frozen CodeLlama + F-01c v24 M2 parity pins -----
MODEL_ID="meta-llama/CodeLlama-7b-hf"
MODEL_REVISION="b462c3c99b077d341db691ec780a33156f3c1472"
MODEL_MODE="codellama-7b"
SEED=2026
GPU_IDS="0,1,2"
WORLD_SIZE=3
PRECISION="bf16"
PER_DEVICE_BATCH=1
GRADIENT_ACCUMULATION=8
GLOBAL_BATCH=24
OPTIMIZER_STEPS=811
SMOKE_STEPS=2
TRAIN_COUNT=19464
VALIDATION_COUNT=512
SOURCE_COUNT=12000
JAVA_TARGET_COUNT=3732
PYTHON_TARGET_COUNT=3732
MAX_SEQ_LENGTH=2048
PHASE_NAMES=(Foundation Adaptation Specialization)
PHASE_STEPS=(270 270 271)
PHASE_SOURCE=(5184 4016 2800)
PHASE_JAVA=(648 1232 1852)
PHASE_PYTHON=(648 1232 1852)
LEARNING_RATE="0.0002"
LORA_RANK=16
LORA_ALPHA=32
LORA_DROPOUT="0.05"
LORA_BIAS="none"
TARGET_MODULES="q_proj,k_proj,v_proj,o_proj"
OPTIMIZER="adamw_torch"
LR_SCHEDULER="cosine"
WEIGHT_DECAY="0.0"
MAX_GRAD_NORM="1.0"
WARMUP_STEPS=20
LOG_STEPS=10
EVAL_STEPS=811
GRADIENT_CHECKPOINTING="true"
GRADIENT_CHECKPOINT_USE_REENTRANT="false"
DDP_FIND_UNUSED_PARAMETERS="false"
TRAIN_ORDER_POLICY="f01e-v4-explicit-global-step-rank-major-v1"
# Exact M2 tokenizer/prompt and smoke-specific schedule. No Python experiment defaults.
HF_HOME="${F02A_HF_HOME:-$HOME/.cache/huggingface}"
MODEL_TYPE="llama"
TOKEN_POLICY="contextual-one-token"
ANSWER_YES=" YES"
ANSWER_NO=" NO"
EXPECTED_YES_TOKEN="22483"
EXPECTED_NO_TOKEN="11698"
PROMPT_PREFIX=$'Determine whether the following two code fragments are code clones. Answer YES if they are clones and NO otherwise.\nLanguage: java\nCode A:\n'
PROMPT_B_SEPARATOR=$'\nCode B:\n'
PROMPT_SUFFIX=$'\nAnswer:'
SMOKE_WARMUP_STEPS=0
SMOKE_EVAL_STEPS=1
SMOKE_LOG_STEPS=1
EVAL_BATCH=1
DATALOADER_WORKERS=0
EVAL_STRATEGY="steps"
SAVE_STRATEGY="steps"
LOAD_BEST_MODEL_AT_END="true"
SAVE_SMOKE_ADAPTER="true"
# v3 full training requires the existing v2 3-GPU real LoRA training smoke proof.
SMOKE_PROOF_RUN="$ROOT/results/f-02a/codellama-7b/2026-1112-161257-1794521577-240365-16213-train-smoke"
EXPECTED_SMOKE_RUN_NAME="2026-1112-161257-1794521577-240365-16213-train-smoke"
FULL_EVAL_STEPS=811
FULL_SAVE_STEPS=811
FULL_LOG_STEPS=10
FULL_WARMUP_STEPS=20
FULL_SAVE_ADAPTER="true"
FULL_GRADIENT_CHECKPOINTING="true"


FROZEN_PREPARED_RUN="$ROOT/results/f-01c/codellama-7b/2026-1112-115525-prepare"
FROZEN_TRAIN_SHA256="e2ec9d2cccc0adb0b73f81f60b3041b5a7f4fa519729f501c1cdc583c1e9b564"
FROZEN_VALIDATION_SHA256="4404f90262c1021917ce31f0c454f87a5a634a0875307b8e84bbd51abefc752a"
FROZEN_MANIFEST_SHA256="149b9439b8eeb2d5dbdf81931cd621b962e02066de0467ab330ce2a7fd933579"
M1_ADAPTER_FILE="$ROOT/results/f-01b/codellama-7b/2026-1111-232648-train/adapter/adapter_model.safetensors"
M1_ADAPTER_SHA256="b8f057ca2e413402738ab7484c0b0108346f4d1c991f00e80ec21508514c78b4"
CURRICULUM_RUN="$ROOT/results/f-01e/codellama-7b/2026-1112-155705-1794520625-239724-11041-prepare"
CURRICULUM_MANIFEST_SHA256="26c0d7cdd0f03eedc123b56add58a27f577c1ebc6c39eaaa7292b92186205128"
CURRICULUM_ORDER_SHA256="ca0ae183c3c8b4c86f2db091c8faf18f66c545e8f6b18335c4b1d361af881566"

# ----- Unique Chicago logs/results -----
LOG_DIR="$ROOT/logs/f-02a/$MODEL_MODE"
RESULTS_DIR="$ROOT/results/f-02a/$MODEL_MODE"
LOG_PREFIX="f-02a-v3"
LOG_TIMESTAMP_FORMAT='+%Y-%m%d-%H%M%S'
mkdir -p "$LOG_DIR" "$RESULTS_DIR"
TS="$(TZ=America/Chicago date "$LOG_TIMESTAMP_FORMAT")"
RID="$(date +%s)-$$-$RANDOM"
LOG_FILE="$LOG_DIR/$LOG_PREFIX-${MODE#--}-$TS-$RID.log"
( set -o noclobber; : > "$LOG_FILE" ) || { echo 'LOG_COLLISION' >&2; exit 2; }
RUN_DIR="$RESULTS_DIR/$TS-$RID-${MODE#--}"
[[ ! -e "$RUN_DIR" ]] || { echo 'RUN_ALREADY_EXISTS' >&2; exit 2; }
START_EPOCH="$(date +%s)"
# All output and Python/Torch stdout/stderr in one automatically created file.
exec > >(tee -a "$LOG_FILE") 2>&1
finish() {
  local rc=$?
  trap - EXIT
  printf '[F-02a v3] FINAL_STATUS=%s EXIT=%s CHICAGO=%s ELAPSED_SECONDS=%s RUN=%s LOG=%s\n' \
    "$([[ $rc -eq 0 ]] && echo PASS || echo FAIL)" "$rc" "$(TZ=America/Chicago date -Is)" \
    "$(( $(date +%s) - START_EPOCH ))" "$RUN_DIR" "$LOG_FILE"
  exit "$rc"
}
trap finish EXIT
step() {
  local name="$1"; shift
  local begin rc
  begin="$(date +%s)"
  printf '[F-02a v3] STEP=%s START CHICAGO=%s\n' "$name" "$(TZ=America/Chicago date -Is)"
  if "$@"; then
    printf '[F-02a v3] STEP=%s PASS CHICAGO=%s ELAPSED_SECONDS=%s\n' "$name" "$(TZ=America/Chicago date -Is)" "$(( $(date +%s) - begin ))"
  else
    rc=$?
    printf '[F-02a v3] STEP=%s FAIL CHICAGO=%s ELAPSED_SECONDS=%s EXIT=%s\n' "$name" "$(TZ=America/Chicago date -Is)" "$(( $(date +%s) - begin ))" "$rc"
    return "$rc"
  fi
}
printf '[F-02a v3] START MODE=%s CHICAGO=%s RUN=%s LOG=%s\n' "$MODE" "$(TZ=America/Chicago date -Is)" "$RUN_DIR" "$LOG_FILE"
printf '[F-02a v3] CONFIG model=%s revision=%s seed=%s gpu_ids=%s batch=%s steps=%s smoke_steps=%s\n' "$MODEL_ID" "$MODEL_REVISION" "$SEED" "$GPU_IDS" "$GLOBAL_BATCH" "$OPTIMIZER_STEPS" "$SMOKE_STEPS"
[[ -f "$PYFILE" ]] || { echo "MISSING $PYFILE" >&2; exit 2; }
export CUDA_VISIBLE_DEVICES="$GPU_IDS" TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=4
ARGS=(
  --hf-home "$HF_HOME" --model-type "$MODEL_TYPE" --token-policy "$TOKEN_POLICY"
  --answer-yes "$ANSWER_YES" --answer-no "$ANSWER_NO"
  --expected-yes-token "$EXPECTED_YES_TOKEN" --expected-no-token "$EXPECTED_NO_TOKEN"
  --prompt-prefix "$PROMPT_PREFIX" --prompt-b-separator "$PROMPT_B_SEPARATOR" --prompt-suffix "$PROMPT_SUFFIX"
  --smoke-warmup-steps "$SMOKE_WARMUP_STEPS" --smoke-eval-steps "$SMOKE_EVAL_STEPS"
  --smoke-log-steps "$SMOKE_LOG_STEPS" --eval-batch "$EVAL_BATCH"
  --dataloader-workers "$DATALOADER_WORKERS" --eval-strategy "$EVAL_STRATEGY"
  --save-strategy "$SAVE_STRATEGY" --load-best-model-at-end "$LOAD_BEST_MODEL_AT_END"
  --save-smoke-adapter "$SAVE_SMOKE_ADAPTER" --learning-rate-float "$LEARNING_RATE"
  --weight-decay-float "$WEIGHT_DECAY" --lora-dropout-float "$LORA_DROPOUT"
  --max-grad-norm-float "$MAX_GRAD_NORM"
  --smoke-proof-run "$SMOKE_PROOF_RUN" --expected-smoke-run-name "$EXPECTED_SMOKE_RUN_NAME" --full-eval-steps "$FULL_EVAL_STEPS"
  --full-save-steps "$FULL_SAVE_STEPS" --full-log-steps "$FULL_LOG_STEPS"
  --full-warmup-steps "$FULL_WARMUP_STEPS" --full-save-adapter "$FULL_SAVE_ADAPTER"
  --full-gradient-checkpointing "$FULL_GRADIENT_CHECKPOINTING"
  --root "$ROOT" --run-dir "$RUN_DIR" --frozen-prepared-run "$FROZEN_PREPARED_RUN"
  --curriculum-run "$CURRICULUM_RUN" --m1-adapter-file "$M1_ADAPTER_FILE" --log-file "$LOG_FILE"
  --frozen-train-sha256 "$FROZEN_TRAIN_SHA256" --frozen-validation-sha256 "$FROZEN_VALIDATION_SHA256"
  --frozen-manifest-sha256 "$FROZEN_MANIFEST_SHA256" --curriculum-manifest-sha256 "$CURRICULUM_MANIFEST_SHA256"
  --curriculum-order-sha256 "$CURRICULUM_ORDER_SHA256" --m1-adapter-sha256 "$M1_ADAPTER_SHA256"
  --model-id "$MODEL_ID" --model-revision "$MODEL_REVISION" --seed "$SEED"
  --gpu-ids "$GPU_IDS" --world-size "$WORLD_SIZE" --precision "$PRECISION"
  --per-device-batch "$PER_DEVICE_BATCH" --gradient-accumulation "$GRADIENT_ACCUMULATION"
  --global-batch "$GLOBAL_BATCH" --optimizer-steps "$OPTIMIZER_STEPS" --smoke-steps "$SMOKE_STEPS"
  --train-count "$TRAIN_COUNT" --validation-count "$VALIDATION_COUNT" --source-count "$SOURCE_COUNT"
  --java-target-count "$JAVA_TARGET_COUNT" --python-target-count "$PYTHON_TARGET_COUNT"
  --max-seq-length "$MAX_SEQ_LENGTH" --phase-names "${PHASE_NAMES[@]}"
  --phase-steps "${PHASE_STEPS[@]}" --phase-source "${PHASE_SOURCE[@]}"
  --phase-java "${PHASE_JAVA[@]}" --phase-python "${PHASE_PYTHON[@]}"
  --learning-rate "$LEARNING_RATE" --lora-rank "$LORA_RANK" --lora-alpha "$LORA_ALPHA"
  --lora-dropout "$LORA_DROPOUT" --lora-bias "$LORA_BIAS" --target-modules "$TARGET_MODULES"
  --optimizer "$OPTIMIZER" --lr-scheduler "$LR_SCHEDULER" --weight-decay "$WEIGHT_DECAY"
  --max-grad-norm "$MAX_GRAD_NORM" --warmup-steps "$WARMUP_STEPS" --log-steps "$LOG_STEPS"
  --eval-steps "$EVAL_STEPS" --gradient-checkpointing "$GRADIENT_CHECKPOINTING"
  --gradient-checkpoint-use-reentrant "$GRADIENT_CHECKPOINT_USE_REENTRANT"
  --ddp-find-unused-parameters "$DDP_FIND_UNUSED_PARAMETERS" --train-order-policy "$TRAIN_ORDER_POLICY"
  --logging-prefix "$LOG_PREFIX" --logging-timestamp-format "$LOG_TIMESTAMP_FORMAT"
)
if [[ "$MODE" == --audit ]]; then
  step FULL_811_STEP_FROZEN_ORDER_AUDIT "$PYTHON_BIN" -u "$PYFILE" audit "${ARGS[@]}"
elif [[ "$MODE" == --smoke ]]; then
  mkdir "$RUN_DIR"
  step THREE_GPU_NCCL_DATALOADER_CONSUMPTION "$PYTHON_BIN" -u -m torch.distributed.run --standalone \
    --nproc_per_node="$WORLD_SIZE" "$PYFILE" worker "${ARGS[@]}"
  step SMOKE_FINALIZE "$PYTHON_BIN" -u "$PYFILE" finalize "${ARGS[@]}"
fi

if [[ "$MODE" == --train-smoke ]]; then
  mkdir "$RUN_DIR"
  step THREE_GPU_REAL_LORA_TRAINING_SMOKE "$PYTHON_BIN" -u -m torch.distributed.run --standalone \
    --nproc_per_node="$WORLD_SIZE" "$PYFILE" train-worker "${ARGS[@]}"
  step TRAINING_SMOKE_FINALIZE "$PYTHON_BIN" -u "$PYFILE" train-finalize "${ARGS[@]}"
fi

if [[ "$MODE" == --train ]]; then
  step VERIFY_APPROVED_TRAINING_SMOKE_AND_FROZEN_PINS "$PYTHON_BIN" -u "$PYFILE" full-preflight "${ARGS[@]}"
  mkdir "$RUN_DIR"
  step FULL_811_STEP_THREE_GPU_CURRICULUM_TRAINING "$PYTHON_BIN" -u -m torch.distributed.run --standalone \
    --nproc_per_node="$WORLD_SIZE" "$PYFILE" full-train-worker "${ARGS[@]}"
  step FULL_TRAINING_FINALIZE "$PYTHON_BIN" -u "$PYFILE" full-train-finalize "${ARGS[@]}"
fi
