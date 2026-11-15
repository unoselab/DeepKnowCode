#!/usr/bin/env bash
# I-01c v2: standalone, shell-configured Gemma 4 BF16 LoRA M1 pipeline.
# No external shell wrapper is invoked. Every experiment-specific parameter is here.
set -Eeuo pipefail
export TZ=America/Chicago
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1
export TOKENIZERS_PARALLELISM=false

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
PYTHON_BIN="python"
ENGINE="$ROOT/src/py2/i-01c-train-source-lora.py"
VERSION="v2"
EXPERIMENT="i-01c"
CONDA_ENV="deepknowcode-gemma"
MODEL_ID="google/gemma-4-31B-it"
MODEL_MODE="gemma-4-31b-it"
MODEL_REVISION="842da3794eaa0b77d5f08bae87a17459d91ff475"
HF_HOME="$HOME/.cache/huggingface"
SNAPSHOT="$HF_HOME/hub/models--google--gemma-4-31B-it/snapshots/$MODEL_REVISION"
EXPECTED_CONFIG_SHA256="e967dd38bc5cfd38bd09a995a7bf4a754075df2b46aba68f7fbb5a791e6d8dd1"
EXPECTED_TOKENIZER_CONFIG_SHA256="9f4fec4b1dc6ecddf8f4a92e9caea5971c0e67d81309f3f9066a2bee8c362633"
CHAT_TEMPLATE_SHA256="ae53464bf3be25802b3a5b37def7fd89667067d7577049b3b2d74c4d8de4c6d4"
TOKENIZER_POLICY="pinned-jinja-autotokenizer-text-only"
TOKENIZER_USE_FAST="true"
TOKENIZER_LOCAL_ONLY="true"
TRUST_REMOTE_CODE="false"
ENABLE_THINKING="false"
BASE_PRECISION="bf16"
FINETUNING_METHOD="lora"
QUANTIZATION="none"
MODEL_AUTO_CLASS="AutoModelForMultimodalLM"
ATTENTION_BACKEND="eager"
LOW_CPU_MEM_USAGE="true"
DEVICE_IDS="0,1,2"
EXPECTED_GPUS=3
DEVICE_MAP="balanced"
MIN_COMPUTE_MAJOR=8
MAX_GPU_PLACEMENT_GIB=39
MAX_CPU_PLACEMENT_GIB=64
MIN_GPU_HEADROOM_GIB=3
MAX_GPU_PEAK_RESERVED_GIB=45
EXPECTED_TORCH="2.15.0"
EXPECTED_TORCH_CUDA="13.2"
EXPECTED_TRANSFORMERS="5.23.0"
EXPECTED_PEFT="0.21.3"
EXPECTED_ACCELERATE="1.15.0"
LORA_TASK_TYPE="CAUSAL_LM"
LORA_TARGET_REGEX='.*language_model\..*\.(q_proj|k_proj|v_proj|o_proj)$'
EXPECTED_LORA_TARGET_MODULES=230
EXPECTED_TRAINABLE_PARAMETERS=45015040
LORA_RANK=16
LORA_ALPHA=32
LORA_DROPOUT="0.05"
LORA_BIAS="none"
GRADIENT_CHECKPOINTING="true"
GRADIENT_CHECKPOINT_USE_REENTRANT="false"
SEED=2026
# Fixed I-01b provenance + immutable prepared file SHA256. No new sampling.
PREPARED_RUN="$ROOT/results/i-01/gemma-4-31b-it/2026-1114-210203-47244-5bypiw"
PREPARED_MANIFEST_SHA256="f9baa2e591c7dbdf2e625180adbea0ab1f8debf1c4b02e1591f0e595cde737c2"
PREPARED_TRAIN_SHA256="40829b9ef2e564272339f77854bbbdf3a8aa77559761f4e77ba13d816d34c0db"
PREPARED_VALIDATION_SHA256="4dc67d82aa13db4ea7781307da55340684426dbb6ae2d06e12d2f0cc085f41ac"
PREPARED_ORDER_SHA256="54dfe7604ea4ca119a0536d840239d0c6dda0bfe94732ad98a7b57d0f4341190"
EXPECTED_PREPARED_FINGERPRINT="f5022121761f775a23f687923c255f672f6a64f42c03737cd4a7f27636b1c0a3"
PREPARED_VERIFICATION_SHA256="002b4fd23d6e7a62dccf7dc1d8414c250786ac2eb0b4c58b3017ced777303a09"
PREPARED_PROVENANCE_SHA256="8b32eddff684f595e69b59394eb5893c7681b48d9623d47972d5f09bdfefb2f7"
I01A_RUN="$ROOT/results/i-01/gemma-4-31b-it/2026-1114-204942-46530-cBCr0Y"
I01A_PREFLIGHT_SHA256="3c08a2aca0a88c2e8cc0034a8439728bf62cea5cf46bce3873c5b860e8b3584b"
I01A_TOKENIZER_SHA256="66809d4fa8e84561829153b31f5100cfc4bfaa8d565af9599235e2b3c5fcbce5"
I01A_LORA_SHA256="f6541e9968ca2ac913bdc9c9a095b59d5b0fd65b77ba1c150d6ca3e40e389612"
I01A_WEIGHTS_SHA256="8eccf85cbf1cfd0fd0508cfd60a2fc71bc9e8d0fe7107c5ed71f19837e4215d3"
EXPECTED_ENGINE_SHA256="65d1c156e346a960c628afd0b067aed627a2f05ff8910cb2fcfc60c7f0a38e3a"
PREPARED_VERIFY_REPORT_STATUS="I01B_V1_INDEPENDENT_VERIFY_PASS_NOT_TRAINED"
SOURCE_CODES="$ROOT/data/training/clean/java_source/data.jsonl"
FROZEN_TRAIN_SHA256="e2ec9d2cccc0adb0b73f81f60b3041b5a7f4fa519729f501c1cdc583c1e9b564"
FROZEN_VALIDATION_SHA256="4404f90262c1021917ce31f0c454f87a5a634a0875307b8e84bbd51abefc752a"
FROZEN_MANIFEST_SHA256="149b9439b8eeb2d5dbdf81931cd621b962e02066de0467ab330ce2a7fd933579"
SOURCE_TRAIN_COUNT=12000
SOURCE_VALIDATION_COUNT=412
SOURCE_TRAIN_POSITIVES=6000
SOURCE_TRAIN_NEGATIVES=6000
SOURCE_VALIDATION_POSITIVES=206
SOURCE_VALIDATION_NEGATIVES=206
MAX_SEQ_LENGTH=2048
EXPECTED_YES_TOKEN_ID=51327
EXPECTED_NO_TOKEN_ID=9424
ANSWER_YES=" YES"
ANSWER_NO=" NO"
SYSTEM_TEXT="You are a code clone classifier. Return exactly YES or NO in the final answer."
USER_PREFIX=$'Determine whether the following two code fragments are code clones. Answer YES if they are clones and NO otherwise.\nLanguage: java\nCode A:\n'
USER_SEPARATOR=$'\nCode B:\n'
USER_SUFFIX=$'\nReturn YES or NO.'
# Single-process model sharding across three GPUs, not data parallelism.
PER_DEVICE_BATCH_SIZE=1
GRADIENT_ACCUMULATION_STEPS=24
MAX_STEPS=500
EXPECTED_EPOCHS=1
OPTIMIZER="adamw_torch"
LEARNING_RATE="0.0002"
WEIGHT_DECAY="0.0"
MAX_GRAD_NORM="1.0"
WARMUP_STEPS=20
LR_SCHEDULER="cosine"
STRESS_STEPS=2
STRESS_MIN_TOKENS=1200
STABILITY_STEPS=8
STABILITY_BALANCED_LABELS="true"
LOG_STEPS=10
PROGRESS_MICROBATCH_EVERY=6
EVAL_STEPS=100
EVAL_BATCH_SIZE=1
SAVE_STEPS=100
SAVE_OPTIMIZER_STATE="true"
LOG_DIR="$ROOT/logs/i-01/$MODEL_MODE"
RESULTS_DIR="$ROOT/results/i-01/$MODEL_MODE"
LOG_PREFIX="i-01c-v2"
LOG_TIMESTAMP_FORMAT='+%Y-%m%d-%H%M%S'
LOG_TEE="true"
LOG_STDERR="true"

MODE="${1:-}"
STRESS_RUN="NONE"
STABILITY_RUN="NONE"
RESUME_CHECKPOINT="NONE"
case "$MODE" in
  --stress) [[ "$#" -eq 1 ]] || { echo 'USAGE: --stress' >&2; exit 2; } ;;
  --stability) [[ "$#" -eq 2 ]] || { echo 'USAGE: --stability STRESS_RUN' >&2; exit 2; }; STRESS_RUN="$2" ;;
  --train) [[ "$#" -eq 3 ]] || { echo 'USAGE: --train STRESS_RUN STABILITY_RUN' >&2; exit 2; }; STRESS_RUN="$2"; STABILITY_RUN="$3" ;;
  --resume) [[ "$#" -eq 4 ]] || { echo 'USAGE: --resume STRESS_RUN STABILITY_RUN CHECKPOINT' >&2; exit 2; }; STRESS_RUN="$2"; STABILITY_RUN="$3"; RESUME_CHECKPOINT="$4" ;;
  *) echo 'USAGE: --stress | --stability STRESS_RUN | --train STRESS_RUN STABILITY_RUN | --resume STRESS_RUN STABILITY_RUN CHECKPOINT' >&2; exit 2 ;;
esac

mkdir -p -- "$LOG_DIR" "$RESULTS_DIR"
START_EPOCH="$(date +%s)"
START_CHICAGO="$(date -Iseconds)"
STAMP="$(date "$LOG_TIMESTAMP_FORMAT")"
RUN_DIR="$(mktemp -d "$RESULTS_DIR/${STAMP}-${BASHPID}-XXXXXX")"
LOG_FILE="$LOG_DIR/${LOG_PREFIX}-${MODE#--}-${STAMP}-${BASHPID}-${RUN_DIR##*-}.log"
( set -o noclobber; : > "$LOG_FILE" ) || { echo "I01C_LOG_COLLISION:$LOG_FILE" >&2; exit 2; }
exec > >(tee -a "$LOG_FILE") 2>&1
export CUDA_VISIBLE_DEVICES="$DEVICE_IDS"

ARGS=(
  --version "$VERSION" --experiment "$EXPERIMENT" "--mode=$MODE"
  --root "$ROOT" --run-dir "$RUN_DIR" --log-file "$LOG_FILE"
  --stress-run "$STRESS_RUN" --stability-run "$STABILITY_RUN"
  --resume-checkpoint "$RESUME_CHECKPOINT"
  --conda-env "$CONDA_ENV" --model-id "$MODEL_ID" --model-mode "$MODEL_MODE"
  --model-revision "$MODEL_REVISION" --hf-home "$HF_HOME" --snapshot "$SNAPSHOT"
  --expected-config-sha256 "$EXPECTED_CONFIG_SHA256"
  --expected-tokenizer-config-sha256 "$EXPECTED_TOKENIZER_CONFIG_SHA256"
  --chat-template-sha256 "$CHAT_TEMPLATE_SHA256"
  --tokenizer-policy "$TOKENIZER_POLICY" --tokenizer-use-fast "$TOKENIZER_USE_FAST"
  --tokenizer-local-only "$TOKENIZER_LOCAL_ONLY" --trust-remote-code "$TRUST_REMOTE_CODE"
  --enable-thinking "$ENABLE_THINKING" --base-precision "$BASE_PRECISION"
  --finetuning-method "$FINETUNING_METHOD" --quantization "$QUANTIZATION"
  --model-auto-class "$MODEL_AUTO_CLASS" --attention-backend "$ATTENTION_BACKEND"
  --low-cpu-mem-usage "$LOW_CPU_MEM_USAGE" --device-ids "$DEVICE_IDS"
  --expected-gpus "$EXPECTED_GPUS" --device-map "$DEVICE_MAP"
  --min-compute-major "$MIN_COMPUTE_MAJOR"
  --max-gpu-placement-gib "$MAX_GPU_PLACEMENT_GIB"
  --max-cpu-placement-gib "$MAX_CPU_PLACEMENT_GIB"
  --min-gpu-headroom-gib "$MIN_GPU_HEADROOM_GIB"
  --max-gpu-peak-reserved-gib "$MAX_GPU_PEAK_RESERVED_GIB"
  --expected-torch "$EXPECTED_TORCH" --expected-torch-cuda "$EXPECTED_TORCH_CUDA"
  --expected-transformers "$EXPECTED_TRANSFORMERS" --expected-peft "$EXPECTED_PEFT"
  --expected-accelerate "$EXPECTED_ACCELERATE"
  --lora-task-type "$LORA_TASK_TYPE" --lora-target-regex "$LORA_TARGET_REGEX"
  --expected-lora-target-modules "$EXPECTED_LORA_TARGET_MODULES"
  --expected-trainable-parameters "$EXPECTED_TRAINABLE_PARAMETERS"
  --lora-rank "$LORA_RANK" --lora-alpha "$LORA_ALPHA"
  --lora-dropout "$LORA_DROPOUT" --lora-bias "$LORA_BIAS"
  --gradient-checkpointing "$GRADIENT_CHECKPOINTING"
  --gradient-checkpoint-use-reentrant "$GRADIENT_CHECKPOINT_USE_REENTRANT"
  --seed "$SEED" --prepared-run "$PREPARED_RUN"
  --prepared-manifest-sha256 "$PREPARED_MANIFEST_SHA256"
  --prepared-train-sha256 "$PREPARED_TRAIN_SHA256"
  --prepared-validation-sha256 "$PREPARED_VALIDATION_SHA256"
  --prepared-order-sha256 "$PREPARED_ORDER_SHA256"
  --expected-prepared-fingerprint "$EXPECTED_PREPARED_FINGERPRINT"
  --prepared-verification-sha256 "$PREPARED_VERIFICATION_SHA256"
  --prepared-provenance-sha256 "$PREPARED_PROVENANCE_SHA256"
  --i01a-run "$I01A_RUN" --i01a-preflight-sha256 "$I01A_PREFLIGHT_SHA256"
  --i01a-tokenizer-sha256 "$I01A_TOKENIZER_SHA256"
  --i01a-lora-sha256 "$I01A_LORA_SHA256"
  --i01a-weights-sha256 "$I01A_WEIGHTS_SHA256"
  --expected-engine-sha256 "$EXPECTED_ENGINE_SHA256"
  --prepared-verify-report-status "$PREPARED_VERIFY_REPORT_STATUS"
  --source-codes "$SOURCE_CODES" --frozen-train-sha256 "$FROZEN_TRAIN_SHA256"
  --frozen-validation-sha256 "$FROZEN_VALIDATION_SHA256"
  --frozen-manifest-sha256 "$FROZEN_MANIFEST_SHA256"
  --source-train-count "$SOURCE_TRAIN_COUNT" --source-validation-count "$SOURCE_VALIDATION_COUNT"
  --source-train-positives "$SOURCE_TRAIN_POSITIVES"
  --source-train-negatives "$SOURCE_TRAIN_NEGATIVES"
  --source-validation-positives "$SOURCE_VALIDATION_POSITIVES"
  --source-validation-negatives "$SOURCE_VALIDATION_NEGATIVES"
  --max-seq-length "$MAX_SEQ_LENGTH" --expected-yes-token-id "$EXPECTED_YES_TOKEN_ID"
  --expected-no-token-id "$EXPECTED_NO_TOKEN_ID"
  --answer-yes "$ANSWER_YES" --answer-no "$ANSWER_NO"
  --system-text "$SYSTEM_TEXT" --user-prefix "$USER_PREFIX"
  --user-separator "$USER_SEPARATOR" --user-suffix "$USER_SUFFIX"
  --per-device-batch-size "$PER_DEVICE_BATCH_SIZE"
  --gradient-accumulation-steps "$GRADIENT_ACCUMULATION_STEPS" --max-steps "$MAX_STEPS"
  --expected-epochs "$EXPECTED_EPOCHS" --optimizer "$OPTIMIZER"
  --learning-rate "$LEARNING_RATE" --weight-decay "$WEIGHT_DECAY"
  --max-grad-norm "$MAX_GRAD_NORM" --warmup-steps "$WARMUP_STEPS"
  --lr-scheduler "$LR_SCHEDULER" --stress-steps "$STRESS_STEPS"
  --stress-min-tokens "$STRESS_MIN_TOKENS" --stability-steps "$STABILITY_STEPS"
  --stability-balanced-labels "$STABILITY_BALANCED_LABELS"
  --log-steps "$LOG_STEPS" --progress-microbatch-every "$PROGRESS_MICROBATCH_EVERY"
  --eval-steps "$EVAL_STEPS" --eval-batch-size "$EVAL_BATCH_SIZE"
  --save-steps "$SAVE_STEPS" --save-optimizer-state "$SAVE_OPTIMIZER_STATE"
  --log-dir "$LOG_DIR" --log-prefix "$LOG_PREFIX"
  --log-timestamp-format "$LOG_TIMESTAMP_FORMAT"
  --log-tee "$LOG_TEE" --log-stderr "$LOG_STDERR"
)

CURRENT_STEP="initialize"
FINALIZED=0
finalize() {
  local rc="$1" status="FAIL" finalizer_rc=0
  (( FINALIZED )) && return "$rc"
  FINALIZED=1
  [[ "$rc" -eq 0 ]] && status="PASS"
  if [[ -f "$ENGINE" ]]; then
    "$PYTHON_BIN" -u "$ENGINE" "${ARGS[@]}" --stage finalize --run-status "$status" --exit-code "$rc" || finalizer_rc="$?"
  else
    finalizer_rc=70
  fi
  if [[ "$finalizer_rc" -ne 0 && "$rc" -eq 0 ]]; then
    rc=70; status="FAIL"
    echo "[I-01c $VERSION] ERROR=FINALIZE_FAILED FINALIZER_EXIT=$finalizer_rc" >&2
  fi
  echo "[I-01c $VERSION] FINAL_STATUS=$status EXIT=$rc STEP=$CURRENT_STEP TOTAL_SECONDS=$(( $(date +%s) - START_EPOCH )) END_CHICAGO=$(date -Iseconds) RUN=$RUN_DIR LOG=$LOG_FILE"
  return "$rc"
}
on_exit() { local rc="$?"; trap - EXIT; finalize "$rc" || rc="$?"; exit "$rc"; }
trap on_exit EXIT
trap 'echo "[I-01c v2] SIGNAL=INT CHICAGO=$(date -Iseconds)"; exit 130' INT
trap 'echo "[I-01c v2] SIGNAL=TERM CHICAGO=$(date -Iseconds)"; exit 143' TERM

echo "[I-01c $VERSION] START MODE=$MODE MODEL=$MODEL_ID REVISION=$MODEL_REVISION CHICAGO=$START_CHICAGO RUN=$RUN_DIR LOG=$LOG_FILE"
echo "[I-01c $VERSION] CONFIG BF16_LORA=true QLORA=false DATASET_TRAIN=$SOURCE_TRAIN_COUNT DATASET_VALIDATION=$SOURCE_VALIDATION_COUNT STEPS=$MAX_STEPS ACCUM=$GRADIENT_ACCUMULATION_STEPS"
[[ -f "$ENGINE" ]] || { echo "I01C_ENGINE_MISSING:$ENGINE" >&2; exit 2; }
[[ "${CONDA_DEFAULT_ENV:-}" == "$CONDA_ENV" ]] || { echo "I01C_WRONG_CONDA_ENV:expected=$CONDA_ENV actual=${CONDA_DEFAULT_ENV:-UNSET}" >&2; exit 2; }
for path in "$STRESS_RUN" "$STABILITY_RUN" "$RESUME_CHECKPOINT"; do
  if [[ "$path" != NONE && ! -d "$path" ]]; then echo "I01C_PREREQUISITE_MISSING:$path" >&2; exit 2; fi
done
step() {
  local name="$1" started rc
  CURRENT_STEP="$name"; started="$(date +%s)"
  echo "[I-01c $VERSION] STEP=$name START CHICAGO=$(date -Iseconds)"
  if "$PYTHON_BIN" -u "$ENGINE" "${ARGS[@]}" --stage "$name"; then
    echo "[I-01c $VERSION] STEP=$name PASS EXIT=0 ELAPSED_SECONDS=$(( $(date +%s) - started )) CHICAGO=$(date -Iseconds)"
  else
    rc="$?"
    echo "[I-01c $VERSION] STEP=$name FAIL EXIT=$rc ELAPSED_SECONDS=$(( $(date +%s) - started )) CHICAGO=$(date -Iseconds)"
    return "$rc"
  fi
}
step preflight
step "${MODE#--}"
exit 0
