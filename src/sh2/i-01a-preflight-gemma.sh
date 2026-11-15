#!/usr/bin/env bash
# I-01a v3: standalone Gemma 4 31B IT BF16 LoRA preflight (NO M1 training).
# Fixed audited snapshot revision; fail if pinned files are missing or mismatched.
# Only Shell defines experiment parameters; all are sent to required Python CLI arguments.
set -Eeuo pipefail
export TZ=America/Chicago
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1
export TOKENIZERS_PARALLELISM=false

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
PYTHON_BIN="python"
ENGINE="$ROOT/src/py2/i-01a-preflight-gemma.py"
VERSION="v3"
EXPERIMENT="i-01a"
MODEL_ID="google/gemma-4-31B-it"
MODEL_MODE="gemma-4-31b-it"
MODEL_TYPE="gemma4"
MODEL_ARCHITECTURE="Gemma4ForConditionalGeneration"
HF_HOME="$HOME/.cache/huggingface"
CACHE_ROOT="$HF_HOME/hub/models--google--gemma-4-31B-it"
REVISION_POLICY="shell-pinned"
MODEL_REVISION="842da3794eaa0b77d5f08bae87a17459d91ff475"
# Obtained from the exact server cache.json of I-01a v2 (2026-11-14 Chicago).
CHAT_TEMPLATE_SHA256="ae53464bf3be25802b3a5b37def7fd89667067d7577049b3b2d74c4d8de4c6d4"
TOKENIZER_POLICY="pinned-jinja-autotokenizer-text-only"
TOKENIZER_USE_FAST="true"
BASE_PRECISION="bf16"
QUANTIZATION="none"
REQUIRE_SHARD_INDEX="true"
MIN_WEIGHT_GIB="40"
DEVICE_IDS="0,1,2"
EXPECTED_GPUS="3"
MIN_COMPUTE_MAJOR="8"
GPU_QUERY_BIN="nvidia-smi"
MIN_GPU_FREE_GIB="4"
MIN_TRANSFORMERS="5.10.1"
MIN_PEFT="0.20.0"
MIN_TORCH="2.6.0"
MIN_ACCELERATE="1.0.0"
MIN_SAFETENSORS="0.4.0"
MIN_PILLOW="10.0.0"
LORA_RANK="16"
LORA_ALPHA="32"
LORA_DROPOUT="0.05"
LORA_BIAS="none"
LORA_TASK_TYPE="CAUSAL_LM"
LOW_CPU_MEM_USAGE="true"
# Gemma 4 includes a vision tower. Restrict LoRA to language_model linear projections.
LORA_TARGET_REGEX='.*language_model\..*\.(q_proj|k_proj|v_proj|o_proj)$'
MIN_TARGET_MODULES="32"
DEVICE_MAP="balanced"
SEED="2026"
LEARNING_RATE="0.0002"
M1_TRAIN_PAIRS="12000"
MIXED_TRAIN_PAIRS="19464"
MIXED_VALIDATION_PAIRS="512"
HELDOUT_PAIRS="7942"
EXPECTED_GT_SHA256="2d171cdee52f6ab8229853b0cc5590b71e1a7dd6919a417dd88fb17d0ec0cb8c"
MAX_SEQ_LENGTH="2048"
SYSTEM_TEXT="You are a code clone classifier. Return exactly YES or NO in the final answer."
USER_PREFIX=$'Determine whether the following two code fragments are code clones. Answer YES if they are clones and NO otherwise.\nLanguage: java\nCode A:\n'
USER_SEPARATOR=$'\nCode B:\n'
USER_SUFFIX=$'\nReturn YES or NO.'
PROBE_CODE_A='int addOne(int x) { return x + 1; }'
PROBE_CODE_B='int increment(int n) { return n + 1; }'
ANSWER_YES=" YES"
ANSWER_NO=" NO"
ENABLE_THINKING="false"
TOKENIZER_LOCAL_ONLY="true"
TRUST_REMOTE_CODE="false"
LOG_DIR="$ROOT/logs/i-01/$MODEL_MODE"
RESULT_DIR="$ROOT/results/i-01/$MODEL_MODE"
LOG_PREFIX="i-01a-v3"
LOG_TIMESTAMP_FORMAT='+%Y-%m%d-%H%M%S'
LOG_TEE="true"
LOG_STDERR="true"
export CUDA_VISIBLE_DEVICES="$DEVICE_IDS"

MODE="${1:-}"
case "$MODE" in
  --audit|--tokenizer|--weights|--lora-probe) ;;
  *) echo "USAGE: bash src/sh2/i-01a-preflight-gemma.sh --audit|--tokenizer|--weights|--lora-probe" >&2; exit 2 ;;
esac
mkdir -p -- "$LOG_DIR" "$RESULT_DIR"
START_EPOCH="$(date +%s)"
START_CHICAGO="$(date -Iseconds)"
STAMP="$(date "$LOG_TIMESTAMP_FORMAT")"
RUN_DIR="$(mktemp -d "$RESULT_DIR/${STAMP}-${BASHPID}-XXXXXX")"
LOG_FILE="$LOG_DIR/${LOG_PREFIX}-${STAMP}-${BASHPID}-${RUN_DIR##*-}.log"
( set -o noclobber; : > "$LOG_FILE" ) || { echo "I01A_LOG_COLLISION:$LOG_FILE" >&2; exit 2; }
# tee captures both stdout and stderr from the controller and Python stages.
exec > >(tee -a "$LOG_FILE") 2>&1
# Even early failures (missing pinned snapshot or invalid revision) receive a unique log.
trap 'rc=$?; echo "[I-01a v3] EARLY_FINAL_STATUS=FAIL EXIT=$rc CHICAGO=$(date -Iseconds) RUN=$RUN_DIR LOG=$LOG_FILE"' EXIT

[[ -f "$ENGINE" ]] || { echo "I01A_ENGINE_MISSING:$ENGINE" >&2; exit 2; }
[[ "$MODEL_REVISION" =~ ^[[:xdigit:]]{40}$ ]] || { echo "I01A_INVALID_REVISION:$MODEL_REVISION" >&2; exit 2; }
SNAPSHOT="$CACHE_ROOT/snapshots/$MODEL_REVISION"
[[ -d "$SNAPSHOT" ]] || { echo "I01A_CACHE_SNAPSHOT_MISSING:$SNAPSHOT" >&2; exit 2; }
if [[ "$MODE" == --lora-probe && "$REVISION_POLICY" != shell-pinned ]]; then
  echo "I01A_LORA_PROBE_REQUIRES_SHELL_PINNED_REVISION: inspect --audit first; publish a new version with fixed MODEL_REVISION and REVISION_POLICY=shell-pinned" >&2
  exit 2
fi



ARGS=(
  --version "$VERSION" --experiment "$EXPERIMENT" "--mode=$MODE"
  --run-dir "$RUN_DIR" --log-file "$LOG_FILE" --root "$ROOT"
  --model-id "$MODEL_ID" --model-mode "$MODEL_MODE"
  --model-type "$MODEL_TYPE" --model-architecture "$MODEL_ARCHITECTURE"
  --hf-home "$HF_HOME" --cache-root "$CACHE_ROOT" --snapshot "$SNAPSHOT"
  --model-revision "$MODEL_REVISION" --revision-policy "$REVISION_POLICY"
  --chat-template-sha256 "$CHAT_TEMPLATE_SHA256" --tokenizer-policy "$TOKENIZER_POLICY"
  --tokenizer-use-fast "$TOKENIZER_USE_FAST"
  --base-precision "$BASE_PRECISION" --quantization "$QUANTIZATION"
  --require-shard-index "$REQUIRE_SHARD_INDEX" --min-weight-gib "$MIN_WEIGHT_GIB"
  --device-ids "$DEVICE_IDS" --expected-gpus "$EXPECTED_GPUS"
  --min-compute-major "$MIN_COMPUTE_MAJOR" --gpu-query-bin "$GPU_QUERY_BIN"
  --min-gpu-free-gib "$MIN_GPU_FREE_GIB"
  --min-transformers "$MIN_TRANSFORMERS" --min-peft "$MIN_PEFT"
  --min-torch "$MIN_TORCH" --min-accelerate "$MIN_ACCELERATE"
  --min-safetensors "$MIN_SAFETENSORS" --min-pillow "$MIN_PILLOW"
  --lora-rank "$LORA_RANK" --lora-alpha "$LORA_ALPHA"
  --lora-dropout "$LORA_DROPOUT" --lora-bias "$LORA_BIAS"
  --lora-task-type "$LORA_TASK_TYPE" --low-cpu-mem-usage "$LOW_CPU_MEM_USAGE"
  --lora-target-regex "$LORA_TARGET_REGEX"
  --min-target-modules "$MIN_TARGET_MODULES" --device-map "$DEVICE_MAP"
  --seed "$SEED" --learning-rate "$LEARNING_RATE"
  --m1-train-pairs "$M1_TRAIN_PAIRS"
  --mixed-train-pairs "$MIXED_TRAIN_PAIRS"
  --mixed-validation-pairs "$MIXED_VALIDATION_PAIRS"
  --heldout-pairs "$HELDOUT_PAIRS" --expected-gt-sha256 "$EXPECTED_GT_SHA256"
  --max-seq-length "$MAX_SEQ_LENGTH"
  --system-text "$SYSTEM_TEXT" --user-prefix "$USER_PREFIX"
  --user-separator "$USER_SEPARATOR" --user-suffix "$USER_SUFFIX"
  --probe-code-a "$PROBE_CODE_A" --probe-code-b "$PROBE_CODE_B"
  --answer-yes "$ANSWER_YES" --answer-no "$ANSWER_NO"
  --enable-thinking "$ENABLE_THINKING" --tokenizer-local-only "$TOKENIZER_LOCAL_ONLY"
  --trust-remote-code "$TRUST_REMOTE_CODE"
  --log-dir "$LOG_DIR" --log-prefix "$LOG_PREFIX"
  --log-timestamp-format "$LOG_TIMESTAMP_FORMAT"
  --log-tee "$LOG_TEE" --log-stderr "$LOG_STDERR"
)

CURRENT_STEP="INIT"
FINALIZED=0
step() {
  local name="$1" start elapsed rc
  CURRENT_STEP="$name"
  start="$(date +%s)"
  echo "[I-01a v3] STEP=$name START CHICAGO=$(date -Iseconds)"
  if "$PYTHON_BIN" "$ENGINE" "${ARGS[@]}" --stage "$name"; then
    elapsed="$(( $(date +%s) - start ))"
    echo "[I-01a v3] STEP=$name PASS EXIT=0 ELAPSED_SECONDS=$elapsed CHICAGO=$(date -Iseconds)"
  else
    rc=$?
    elapsed="$(( $(date +%s) - start ))"
    echo "[I-01a v3] STEP=$name FAIL EXIT=$rc ELAPSED_SECONDS=$elapsed CHICAGO=$(date -Iseconds)"
    return "$rc"
  fi
}
finalize() {
  local rc="$1" status="FAIL"
  (( FINALIZED )) && return 0
  FINALIZED=1
  [[ "$rc" -eq 0 ]] && status="PASS"
  "$PYTHON_BIN" "$ENGINE" "${ARGS[@]}" --stage finalize --run-status "$status" --exit-code "$rc" || true
  echo "[I-01a v3] FINAL_STATUS=$status EXIT=$rc TOTAL_SECONDS=$(( $(date +%s) - START_EPOCH )) END_CHICAGO=$(date -Iseconds) RUN=$RUN_DIR LOG=$LOG_FILE"
}
on_exit() {
  local rc="$?"
  trap - EXIT
  finalize "$rc"
  exit "$rc"
}
trap on_exit EXIT
trap 'echo "[I-01a v3] SIGNAL=INT CHICAGO=$(date -Iseconds)"; exit 130' INT
trap 'echo "[I-01a v3] SIGNAL=TERM CHICAGO=$(date -Iseconds)"; exit 143' TERM

echo "[I-01a v3] START MODE=$MODE MODEL=$MODEL_ID REVISION=$MODEL_REVISION REVISION_POLICY=$REVISION_POLICY CHICAGO=$START_CHICAGO RUN=$RUN_DIR LOG=$LOG_FILE"
echo "[I-01a v3] METHOD=BF16_LORA QUANTIZATION=none TRAINING=NO REVISION=PINNED TOKENIZER_POLICY=$TOKENIZER_POLICY"
step cache
step gpu
step environment
case "$MODE" in
  --audit) ;;
  --tokenizer) step tokenizer ;;
  --weights) step weights ;;
  --lora-probe) step tokenizer; step weights; step lora_probe ;;
esac
exit 0
