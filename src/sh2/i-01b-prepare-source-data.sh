#!/usr/bin/env bash
# I-01b v1: standalone, read-only-on-inputs Gemma M1 frozen Java source preparation.
# Never calls another shell wrapper; never trains/loads model weights or changes existing artifacts.
set -Eeuo pipefail
export TZ=America/Chicago
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export HF_HUB_DISABLE_TELEMETRY=1

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
PYTHON_BIN="python"
ENGINE="$ROOT/src/py2/i-01b-prepare-source-data.py"
VERSION="v1"
EXPERIMENT="i-01b"
CONDA_ENV="deepknowcode-gemma"
MODEL_ID="google/gemma-4-31B-it"
MODEL_MODE="gemma-4-31b-it"
MODEL_REVISION="842da3794eaa0b77d5f08bae87a17459d91ff475"
HF_HOME="$HOME/.cache/huggingface"
SNAPSHOT="$HF_HOME/hub/models--google--gemma-4-31B-it/snapshots/$MODEL_REVISION"
CHAT_TEMPLATE_SHA256="ae53464bf3be25802b3a5b37def7fd89667067d7577049b3b2d74c4d8de4c6d4"
EXPECTED_CONFIG_SHA256="e967dd38bc5cfd38bd09a995a7bf4a754075df2b46aba68f7fbb5a791e6d8dd1"
EXPECTED_TOKENIZER_CONFIG_SHA256="9f4fec4b1dc6ecddf8f4a92e9caea5971c0e67d81309f3f9066a2bee8c362633"
TOKENIZER_POLICY="pinned-jinja-autotokenizer-text-only"
TOKENIZER_USE_FAST="true"
TOKENIZER_LOCAL_ONLY="true"
TRUST_REMOTE_CODE="false"
ENABLE_THINKING="false"
BASE_PRECISION="bf16"
FINETUNING_METHOD="lora"
QUANTIZATION="none"
LORA_RANK=16
LORA_ALPHA=32
LORA_DROPOUT="0.05"
LORA_BIAS="none"
LORA_TASK_TYPE="CAUSAL_LM"
LORA_TARGET_REGEX='.*language_model\..*\.(q_proj|k_proj|v_proj|o_proj)$'
EXPECTED_LORA_TARGET_MODULES=230
EXPECTED_TRAINABLE_PARAMETERS=45015040
DEVICE_IDS="0,1,2"
EXPECTED_GPUS=3
DEVICE_MAP="balanced"
SEED=2026
# Approved fixed M1 budget from G-01d: 1 process x 1 batch x 24 accumulation x 500 steps.
PER_DEVICE_BATCH_SIZE=1
GRADIENT_ACCUMULATION_STEPS=24
MAX_STEPS=500
EXPECTED_EPOCHS=1
LEARNING_RATE="0.0002"
MAX_SEQ_LENGTH=2048
AUDIT_PROGRESS_EVERY=1000
EXPECTED_YES_TOKEN_ID=51327
EXPECTED_NO_TOKEN_ID=9424
ANSWER_YES=" YES"
ANSWER_NO=" NO"
SYSTEM_TEXT="You are a code clone classifier. Return exactly YES or NO in the final answer."
USER_PREFIX=$'Determine whether the following two code fragments are code clones. Answer YES if they are clones and NO otherwise.\nLanguage: java\nCode A:\n'
USER_SEPARATOR=$'\nCode B:\n'
USER_SUFFIX=$'\nReturn YES or NO.'
PROBE_CODE_A='int addOne(int x) { return x + 1; }'
PROBE_CODE_B='int increment(int n) { return n + 1; }'
# Successful I-01a v3 BF16 backward proof supplied by the operator; immutable evidence.
I01A_RUN="$ROOT/results/i-01/gemma-4-31b-it/2026-1114-204942-46530-cBCr0Y"
I01A_PREFLIGHT_SHA256="3c08a2aca0a88c2e8cc0034a8439728bf62cea5cf46bce3873c5b860e8b3584b"
I01A_TOKENIZER_SHA256="66809d4fa8e84561829153b31f5100cfc4bfaa8d565af9599235e2b3c5fcbce5"
I01A_LORA_SHA256="f6541e9968ca2ac913bdc9c9a095b59d5b0fd65b77ba1c150d6ca3e40e389612"
I01A_WEIGHTS_SHA256="8eccf85cbf1cfd0fd0508cfd60a2fc71bc9e8d0fe7107c5ed71f19837e4215d3"
# Exact F-01c frozen mixed source from the previously completed CodeLlama and GPT-OSS stages.
FROZEN_PREPARED="$ROOT/results/f-01c/codellama-7b/2026-1112-115525-prepare/prepared"
FROZEN_TRAIN_SHA256="e2ec9d2cccc0adb0b73f81f60b3041b5a7f4fa519729f501c1cdc583c1e9b564"
FROZEN_VALIDATION_SHA256="4404f90262c1021917ce31f0c454f87a5a634a0875307b8e84bbd51abefc752a"
FROZEN_MANIFEST_SHA256="149b9439b8eeb2d5dbdf81931cd621b962e02066de0467ab330ce2a7fd933579"
SOURCE_CODES="$ROOT/data/training/clean/java_source/data.jsonl"
FROZEN_TRAIN_COUNT=19464
FROZEN_VALIDATION_COUNT=512
SOURCE_TRAIN_COUNT=12000
SOURCE_VALIDATION_COUNT=412
SOURCE_TRAIN_POSITIVES=6000
SOURCE_TRAIN_NEGATIVES=6000
SOURCE_VALIDATION_POSITIVES=206
SOURCE_VALIDATION_NEGATIVES=206
# Cross-model ground truth: verify G-01c's exact previously approved source Pair IDs/Order.
G01C_PREPARED="$ROOT/results/g-01c/gpt-oss-120b/2026-1112-235431-1794549271-271004-23954-prepare/prepared"
G01C_MANIFEST_SHA256="d61894e8c55a3aa31f9bdeff7d2003fee50ff993e85a2cac9d005cb3f5049449"
G01C_TRAIN_SHA256="3cf5b22e5f74191662e3607411fe19c1077c81cb180cc0bd0a11b389dd385759"
G01C_VALIDATION_SHA256="98856974b9f8c60df9f2dd22dc355c1bb6f4468e941bbda97e87a4c60b26827e"
G01C_ORDER_SHA256="d2dcb384da3b22622abdb1361c15e172a7b66648582f64a75e2d180da3f9de55"
# Exact values above are verified during package build against known G-01d references.
LOG_DIR="$ROOT/logs/i-01/$MODEL_MODE"
RESULTS_DIR="$ROOT/results/i-01/$MODEL_MODE"
LOG_PREFIX="i-01b-v1"
LOG_TIMESTAMP_FORMAT='+%Y-%m%d-%H%M%S'
LOG_TEE="true"
LOG_STDERR="true"

MODE="${1:-}"
VERIFY_SOURCE="NONE"
case "$MODE" in
  --prepare) [[ "$#" -eq 1 ]] || { echo 'USAGE: --prepare' >&2; exit 2; } ;;
  --verify) [[ "$#" -eq 2 ]] || { echo 'USAGE: --verify <prepared_run_dir>' >&2; exit 2; }; VERIFY_SOURCE="$2" ;;
  *) echo 'USAGE: bash src/sh2/i-01b-prepare-source-data.sh --prepare | --verify <prepared_run_dir>' >&2; exit 2 ;;
esac

mkdir -p -- "$LOG_DIR" "$RESULTS_DIR"
START_EPOCH="$(date +%s)"
STAMP="$(date "$LOG_TIMESTAMP_FORMAT")"
START_CHICAGO="$(date -Iseconds)"
RUN_DIR="$(mktemp -d "$RESULTS_DIR/${STAMP}-${BASHPID}-XXXXXX")"
LOG_FILE="$LOG_DIR/${LOG_PREFIX}-${MODE#--}-${STAMP}-${BASHPID}-${RUN_DIR##*-}.log"
( set -o noclobber; : > "$LOG_FILE" ) || { echo "I01B_LOG_COLLISION:$LOG_FILE" >&2; exit 2; }
exec > >(tee -a "$LOG_FILE") 2>&1

ARGS=(
  --version "$VERSION" --experiment "$EXPERIMENT" "--mode=$MODE"
  --root "$ROOT" --run-dir "$RUN_DIR" --log-file "$LOG_FILE" --verify-source "$VERIFY_SOURCE"
  --conda-env "$CONDA_ENV" --model-id "$MODEL_ID" --model-mode "$MODEL_MODE"
  --model-revision "$MODEL_REVISION" --snapshot "$SNAPSHOT" --hf-home "$HF_HOME"
  --chat-template-sha256 "$CHAT_TEMPLATE_SHA256"
  --expected-config-sha256 "$EXPECTED_CONFIG_SHA256"
  --expected-tokenizer-config-sha256 "$EXPECTED_TOKENIZER_CONFIG_SHA256"
  --tokenizer-policy "$TOKENIZER_POLICY" --tokenizer-use-fast "$TOKENIZER_USE_FAST"
  --tokenizer-local-only "$TOKENIZER_LOCAL_ONLY" --trust-remote-code "$TRUST_REMOTE_CODE"
  --enable-thinking "$ENABLE_THINKING" --base-precision "$BASE_PRECISION"
  --finetuning-method "$FINETUNING_METHOD" --quantization "$QUANTIZATION"
  --lora-rank "$LORA_RANK" --lora-alpha "$LORA_ALPHA"
  --lora-dropout "$LORA_DROPOUT" --lora-bias "$LORA_BIAS"
  --lora-task-type "$LORA_TASK_TYPE" --lora-target-regex "$LORA_TARGET_REGEX"
  --expected-lora-target-modules "$EXPECTED_LORA_TARGET_MODULES"
  --expected-trainable-parameters "$EXPECTED_TRAINABLE_PARAMETERS"
  --device-ids "$DEVICE_IDS" --expected-gpus "$EXPECTED_GPUS" --device-map "$DEVICE_MAP"
  --seed "$SEED" --per-device-batch-size "$PER_DEVICE_BATCH_SIZE"
  --gradient-accumulation-steps "$GRADIENT_ACCUMULATION_STEPS"
  --max-steps "$MAX_STEPS" --expected-epochs "$EXPECTED_EPOCHS"
  --learning-rate "$LEARNING_RATE" --max-seq-length "$MAX_SEQ_LENGTH"
  --audit-progress-every "$AUDIT_PROGRESS_EVERY"
  --expected-yes-token-id "$EXPECTED_YES_TOKEN_ID" --expected-no-token-id "$EXPECTED_NO_TOKEN_ID"
  --answer-yes "$ANSWER_YES" --answer-no "$ANSWER_NO"
  --system-text "$SYSTEM_TEXT" --user-prefix "$USER_PREFIX"
  --user-separator "$USER_SEPARATOR" --user-suffix "$USER_SUFFIX"
  --probe-code-a "$PROBE_CODE_A" --probe-code-b "$PROBE_CODE_B"
  --i01a-run "$I01A_RUN" --i01a-preflight-sha256 "$I01A_PREFLIGHT_SHA256"
  --i01a-tokenizer-sha256 "$I01A_TOKENIZER_SHA256" --i01a-lora-sha256 "$I01A_LORA_SHA256"
  --i01a-weights-sha256 "$I01A_WEIGHTS_SHA256"
  --frozen-prepared "$FROZEN_PREPARED" --frozen-train-sha256 "$FROZEN_TRAIN_SHA256"
  --frozen-validation-sha256 "$FROZEN_VALIDATION_SHA256"
  --frozen-manifest-sha256 "$FROZEN_MANIFEST_SHA256" --source-codes "$SOURCE_CODES"
  --frozen-train-count "$FROZEN_TRAIN_COUNT" --frozen-validation-count "$FROZEN_VALIDATION_COUNT"
  --source-train-count "$SOURCE_TRAIN_COUNT" --source-validation-count "$SOURCE_VALIDATION_COUNT"
  --source-train-positives "$SOURCE_TRAIN_POSITIVES" --source-train-negatives "$SOURCE_TRAIN_NEGATIVES"
  --source-validation-positives "$SOURCE_VALIDATION_POSITIVES" --source-validation-negatives "$SOURCE_VALIDATION_NEGATIVES"
  --g01c-prepared "$G01C_PREPARED" --g01c-manifest-sha256 "$G01C_MANIFEST_SHA256"
  --g01c-train-sha256 "$G01C_TRAIN_SHA256"
  --g01c-validation-sha256 "$G01C_VALIDATION_SHA256"
  --g01c-order-sha256 "$G01C_ORDER_SHA256"
  --log-dir "$LOG_DIR" --log-prefix "$LOG_PREFIX"
  --log-timestamp-format "$LOG_TIMESTAMP_FORMAT"
  --log-tee "$LOG_TEE" --log-stderr "$LOG_STDERR"
)

CURRENT_STEP="INITIALIZE"
FINALIZED=0
finalize() {
  local rc="$1" status="FAIL"
  (( FINALIZED )) && return 0
  FINALIZED=1
  [[ "$rc" -eq 0 ]] && status="PASS"
  "$PYTHON_BIN" -u "$ENGINE" "${ARGS[@]}" --stage finalize --run-status "$status" --exit-code "$rc" || true
  echo "[I-01b $VERSION] FINAL_STATUS=$status EXIT=$rc TOTAL_SECONDS=$(( $(date +%s) - START_EPOCH )) END_CHICAGO=$(date -Iseconds) RUN=$RUN_DIR LOG=$LOG_FILE"
}
on_exit() {
  local rc="$?"
  trap - EXIT
  finalize "$rc"
  exit "$rc"
}
trap on_exit EXIT
trap 'echo "[I-01b v1] SIGNAL=INT CHICAGO=$(date -Iseconds)"; exit 130' INT
trap 'echo "[I-01b v1] SIGNAL=TERM CHICAGO=$(date -Iseconds)"; exit 143' TERM

echo "[I-01b $VERSION] START MODE=$MODE MODEL=$MODEL_ID REVISION=$MODEL_REVISION TRAINING=NO CHICAGO=$START_CHICAGO RUN=$RUN_DIR LOG=$LOG_FILE"
[[ -f "$ENGINE" ]] || { echo "I01B_ENGINE_MISSING:$ENGINE" >&2; exit 2; }
[[ "${CONDA_DEFAULT_ENV:-}" == "$CONDA_ENV" ]] || { echo "I01B_WRONG_CONDA_ENV:expected=$CONDA_ENV actual=${CONDA_DEFAULT_ENV:-UNSET}" >&2; exit 2; }
if [[ "$MODE" == --verify ]]; then
  [[ -d "$VERIFY_SOURCE" ]] || { echo "I01B_VERIFY_SOURCE_MISSING:$VERIFY_SOURCE" >&2; exit 2; }
fi
step() {
  local name="$1" start rc
  CURRENT_STEP="$name"; start="$(date +%s)"
  echo "[I-01b $VERSION] STEP=$name START CHICAGO=$(date -Iseconds)"
  if "$PYTHON_BIN" -u "$ENGINE" "${ARGS[@]}" --stage "$name"; then
    echo "[I-01b $VERSION] STEP=$name PASS EXIT=0 ELAPSED_SECONDS=$(( $(date +%s) - start )) CHICAGO=$(date -Iseconds)"
  else
    rc=$?
    echo "[I-01b $VERSION] STEP=$name FAIL EXIT=$rc ELAPSED_SECONDS=$(( $(date +%s) - start )) CHICAGO=$(date -Iseconds)"
    return "$rc"
  fi
}
step provenance
step frozen
step tokenizer
if [[ "$MODE" == --prepare ]]; then step prepare; fi
step verify
exit 0
