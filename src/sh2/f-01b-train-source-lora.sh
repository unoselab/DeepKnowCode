#!/usr/bin/env bash
# F-01b v1: standalone multi-LLM M1 source-only LoRA entrypoint.
# All experiment-specific configuration lives in THIS shell script.
# No calls to A-11/E-01b/D-03a shell wrappers.

# # Step 1 — Preflight
# bash src/sh2/f-01b-train-source-lora.sh --preflight

# # Step 2 — Dataset Preparation
# bash src/sh2/f-01b-train-source-lora.sh --prepare

# # Step 3 — 3-GPU DDP Smoke Test
# bash src/sh2/f-01b-train-source-lora.sh --smoke

# # Step 4 — Full M1 Training
# bash src/sh2/f-01b-train-source-lora.sh --train


set -Eeuo pipefail
export TZ=America/Chicago
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
PYTHON_BIN="${F01B_PYTHON_BIN:-python}"
PYFILE="$ROOT/src/py2/f-01b-train-source-lora.py"
[[ -f "$PYFILE" ]] || { echo "Missing Python engine: $PYFILE" >&2; exit 2; }

# -------- 1. Model-specific registry (ONLY model config varies) --------
MODEL_MODE="${F01B_MODEL_MODE:-codellama-7b}"
case "$MODEL_MODE" in
  starcoder2-7b)
    MODEL_ID="bigcode/starcoder2-7b"
    MODEL_REVISION="bb9afde76d7945da5745592525db122d4d729eb1"
    MODEL_FAMILY="starcoder2"
    EXPECTED_MODEL_TYPE="starcoder2"
    TOKEN_POLICY="standalone-one-token"
    EXPECTED_YES_TOKEN="derive"
    EXPECTED_NO_TOKEN="derive"
    TARGET_MODULES="q_proj,k_proj,v_proj,o_proj"
    ;;
  starcoder2-15b)
    MODEL_ID="bigcode/starcoder2-15b"
    MODEL_REVISION="46d44742909c03ac8cee08eb03fdebce02e193ec"
    MODEL_FAMILY="starcoder2"
    EXPECTED_MODEL_TYPE="starcoder2"
    TOKEN_POLICY="standalone-one-token"
    EXPECTED_YES_TOKEN="derive"
    EXPECTED_NO_TOKEN="derive"
    TARGET_MODULES="q_proj,k_proj,v_proj,o_proj"
    ;;
  codellama-7b)
    MODEL_ID="meta-llama/CodeLlama-7b-hf"
    MODEL_REVISION="b462c3c99b077d341db691ec780a33156f3c1472"
    MODEL_FAMILY="llama"
    EXPECTED_MODEL_TYPE="llama"
    TOKEN_POLICY="contextual-one-token"
    EXPECTED_YES_TOKEN="22483"
    EXPECTED_NO_TOKEN="11698"
    TARGET_MODULES="q_proj,k_proj,v_proj,o_proj"
    ;;
  *) echo "ERROR: unsupported MODEL_MODE=$MODEL_MODE" >&2; exit 2 ;;
esac

# -------- 2. Common dataset, tokenizer, selection --------
SOURCE_CORPUS="java_source"
SOURCE_DIR="$ROOT/data/training/clean/$SOURCE_CORPUS"
STAGING_MANIFEST="$ROOT/data/training/manifest.json"
GROUND_TRUTH="$ROOT/data/evaluation/ground_truth/python/verified_pairs.jsonl"
EXPECTED_GROUND_TRUTH_SHA256="2d171cdee52f6ab8229853b0cc5590b71e1a7dd6919a417dd88fb17d0ec0cb8c"
HELDOUT_JAVA_CODES="$ROOT/data/evaluation/code/java.jsonl"
HELDOUT_PYTHON_CODES="$ROOT/data/evaluation/code/python.jsonl"
VERIFY_HOLDOUT_OVERLAP="strict"
MAX_OVERLAP_EXAMPLES=8
EXPECTED_A10_TEST_PAIRS=39002
A10_AUDIT_FILE="${F01B_A10_AUDIT_FILE:-}"
A10_DISCOVERY="newest-valid"  # Explicit fallback policy; resolved audit hash is pinned in manifest.
SAMPLE_POLICY="balanced"
SEED=2026
MAX_SEQ_LENGTH=2048
TRAIN_EXAMPLES=12000
VAL_EXAMPLES=512
SAMPLING_RNG_LABEL_STRIDE=991
SAMPLING_OVERSAMPLE_MIN=64
SAMPLING_OVERSAMPLE_DIVISOR=4
SAMPLING_VAL_SEED_OFFSET=1234567
SAMPLING_ORDER_SEED_OFFSET=97813
ANSWER_YES=" YES"
ANSWER_NO=" NO"
PROMPT_PREFIX=$'Determine whether the following two code fragments are code clones. Answer YES if they are clones and NO otherwise.\nLanguage: java\nCode A:\n'
PROMPT_B_SEPARATOR=$'\nCode B:\n'
PROMPT_SUFFIX=$'\nAnswer:'
PROBE_CODE_A='public int a(int x) { return x + 1; }'
PROBE_CODE_B='public int b(int x) { return x + 1; }'

# -------- 3. Common LoRA/DDP/Trainer controls --------
GPU_IDS="${F01B_GPU_IDS:-0,1,2}"
EXPECTED_WORLD_SIZE=3
MIN_VRAM_GIB=16
PRECISION="bf16"
LEARNING_RATE="0.0002"
LORA_RANK=16
LORA_ALPHA=32
LORA_DROPOUT="0.05"
LORA_BIAS="none"
PER_DEVICE_BATCH_SIZE=1
PER_DEVICE_EVAL_BATCH_SIZE=1
GRADIENT_ACCUMULATION_STEPS=8
MAX_STEPS=500
EVAL_STEPS=100
LOG_STEPS=10
WARMUP_STEPS=20
LR_SCHEDULER="cosine"
WEIGHT_DECAY="0.0"
OPTIMIZER="adamw_torch"
GRADIENT_CHECKPOINTING="true"
GRADIENT_CHECKPOINT_USE_REENTRANT="false"
DDP_FIND_UNUSED_PARAMETERS="false"
DATALOADER_WORKERS=0
LOGGING_FIRST_STEP="true"
REPORT_TO="none"
SAVE_STRATEGY="steps"
EVAL_STRATEGY="steps"
SAVE_TOTAL_LIMIT=0  # 0 = no checkpoint pruning; never discard historical run artifacts.
LOAD_BEST_MODEL_AT_END="true"
METRIC_FOR_BEST_MODEL="eval_loss"
GREATER_IS_BETTER="false"
REMOVE_UNUSED_COLUMNS="false"
SAVE_SAFETENSORS="true"
DISABLE_TQDM="false"
EVAL_ACCUMULATION_STEPS=1
USE_CACHE="false"
TRAIN_ORDER="shuffled"
MAX_GRAD_NORM="1.0"
LABEL_SMOOTHING_FACTOR="0.0"
DATALOADER_DROP_LAST="false"
FULL_DETERMINISM="false"
BF16_FULL_EVAL="false"

# -------- 4. Modes (smoke configuration is explicit in Shell) --------
MODE="${1:---preflight}"
if (( $# )); then shift; fi
VERIFY_RUN=""
case "$MODE" in
  --preflight) PURPOSE="preflight" ;;
  --prepare) PURPOSE="prepare" ;;
  --smoke) PURPOSE="smoke" ;;
  --train) PURPOSE="train" ;;
  --verify)
    [[ $# -eq 1 ]] || { echo 'Usage: --verify /path/to/completed-run' >&2; exit 2; }
    VERIFY_RUN="$1"; shift
    [[ -d "$VERIFY_RUN" ]] || { echo "Missing verify run: $VERIFY_RUN" >&2; exit 2; }
    VERIFY_RUN="$(cd "$VERIFY_RUN" && pwd -P)"
    case "$(basename "$VERIFY_RUN")" in
      *-smoke) PURPOSE="smoke" ;;
      *-train) PURPOSE="train" ;;
      *) echo 'Run directory must end with -smoke or -train' >&2; exit 2 ;;
    esac
    ;;
  *) echo 'Usage: --preflight|--prepare|--smoke|--train|--verify RUN_DIR' >&2; exit 2 ;;
esac
[[ $# -eq 0 ]] || { echo 'Unexpected command arguments' >&2; exit 2; }
if [[ "$PURPOSE" == "smoke" ]]; then
  TRAIN_EXAMPLES=96
  VAL_EXAMPLES=32
  MAX_STEPS=2
  EVAL_STEPS=1
  LOG_STEPS=1
  WARMUP_STEPS=0
fi
IFS=',' read -r -a GPU_ID_ARRAY <<< "$GPU_IDS"
[[ "${#GPU_ID_ARRAY[@]}" -eq "$EXPECTED_WORLD_SIZE" ]] || { echo 'GPU list length differs from expected world' >&2; exit 2; }
declare -A SEEN_GPU=()
for id in "${GPU_ID_ARRAY[@]}"; do
  [[ "$id" =~ ^[0-9]+$ ]] || { echo "Invalid GPU ID $id" >&2; exit 2; }
  [[ -z "${SEEN_GPU[$id]+exists}" ]] || { echo "Duplicate GPU ID $id" >&2; exit 2; }
  SEEN_GPU[$id]=1
done
export CUDA_VISIBLE_DEVICES="$GPU_IDS"
export HF_HOME="${F01B_HF_HOME:-$HOME/.cache/huggingface}"
export TOKENIZERS_PARALLELISM="false"
export OMP_NUM_THREADS="${F01B_OMP_NUM_THREADS:-4}"

# -------- 5. Chicago timestamp, unique no-clobber logs/results --------
LOG_DIR="$ROOT/logs/f-01b/$MODEL_MODE"
RESULTS_DIR="$ROOT/results/f-01b/$MODEL_MODE"
LOG_PREFIX="f-01b-v1"
LOG_TIMESTAMP_FORMAT='+%Y-%m%d-%H%M%S'
CHICAGO_TIMESTAMP="$(TZ=America/Chicago date "$LOG_TIMESTAMP_FORMAT")"
mkdir -p "$LOG_DIR" "$RESULTS_DIR"
LOG_FILE="$LOG_DIR/$LOG_PREFIX-$MODEL_MODE-${MODE#--}-$CHICAGO_TIMESTAMP.log"
if ! ( set -o noclobber; : > "$LOG_FILE" ) 2>/dev/null; then
  echo "ERROR: Refusing to overwrite existing log: $LOG_FILE" >&2
  exit 2
fi
if [[ "$MODE" == --verify ]]; then
  RUN_DIR="$VERIFY_RUN"
else
  RUN_DIR="$RESULTS_DIR/$CHICAGO_TIMESTAMP-$PURPOSE"
  [[ ! -e "$RUN_DIR" ]] || { echo "Run already exists: $RUN_DIR" | tee -a "$LOG_FILE" >&2; exit 2; }
fi

PREPARED_DIR="$RUN_DIR/prepared"
ADAPTER_DIR="$RUN_DIR/adapter"
CHECKPOINT_DIR="$RUN_DIR/checkpoints"

# Every argument passed explicitly, no experiment defaults inside Python.
COMMON=(
  --root "$ROOT" --run-dir "$RUN_DIR" --purpose "$PURPOSE"
  --prepared-dir "$PREPARED_DIR" --adapter-dir "$ADAPTER_DIR" --checkpoint-dir "$CHECKPOINT_DIR"
  --gpu-ids "$GPU_IDS" --log-file "$LOG_FILE" --log-dir "$LOG_DIR"
  --log-prefix "$LOG_PREFIX" --log-timestamp "$CHICAGO_TIMESTAMP"
  --log-timestamp-format "$LOG_TIMESTAMP_FORMAT"
  --model-mode "$MODEL_MODE" --model-id "$MODEL_ID" --model-revision "$MODEL_REVISION"
  --model-family "$MODEL_FAMILY" --expected-model-type "$EXPECTED_MODEL_TYPE"
  --token-policy "$TOKEN_POLICY" --expected-yes-token "$EXPECTED_YES_TOKEN"
  --expected-no-token "$EXPECTED_NO_TOKEN" --target-modules "$TARGET_MODULES"
  --source-corpus "$SOURCE_CORPUS" --source-dir "$SOURCE_DIR"
  --staging-manifest "$STAGING_MANIFEST" --ground-truth "$GROUND_TRUTH"
  --expected-ground-truth-sha256 "$EXPECTED_GROUND_TRUTH_SHA256"
  --heldout-java-codes "$HELDOUT_JAVA_CODES" --heldout-python-codes "$HELDOUT_PYTHON_CODES"
  --verify-holdout-overlap "$VERIFY_HOLDOUT_OVERLAP" --max-overlap-examples "$MAX_OVERLAP_EXAMPLES"
  --a10-audit-file "$A10_AUDIT_FILE" --a10-discovery "$A10_DISCOVERY"
  --expected-a10-test-pairs "$EXPECTED_A10_TEST_PAIRS"
  --sample-policy "$SAMPLE_POLICY" --seed "$SEED"
  --max-train-examples "$TRAIN_EXAMPLES" --max-val-examples "$VAL_EXAMPLES"
  --max-seq-length "$MAX_SEQ_LENGTH"
  --sample-rng-label-stride "$SAMPLING_RNG_LABEL_STRIDE"
  --sample-oversample-min "$SAMPLING_OVERSAMPLE_MIN"
  --sample-oversample-divisor "$SAMPLING_OVERSAMPLE_DIVISOR"
  --sample-val-seed-offset "$SAMPLING_VAL_SEED_OFFSET"
  --sample-order-seed-offset "$SAMPLING_ORDER_SEED_OFFSET"
  --answer-yes "$ANSWER_YES" --answer-no "$ANSWER_NO"
  --prompt-prefix "$PROMPT_PREFIX" --prompt-b-separator "$PROMPT_B_SEPARATOR"
  --prompt-suffix "$PROMPT_SUFFIX" --probe-code-a "$PROBE_CODE_A" --probe-code-b "$PROBE_CODE_B"
  --hf-home "$HF_HOME" --precision "$PRECISION" --min-vram-gib "$MIN_VRAM_GIB"
  --expected-world-size "$EXPECTED_WORLD_SIZE" --learning-rate "$LEARNING_RATE"
  --lora-rank "$LORA_RANK" --lora-alpha "$LORA_ALPHA" --lora-dropout "$LORA_DROPOUT"
  --lora-bias "$LORA_BIAS" --per-device-batch-size "$PER_DEVICE_BATCH_SIZE"
  --per-device-eval-batch-size "$PER_DEVICE_EVAL_BATCH_SIZE"
  --gradient-accumulation-steps "$GRADIENT_ACCUMULATION_STEPS"
  --max-steps "$MAX_STEPS" --eval-steps "$EVAL_STEPS" --log-steps "$LOG_STEPS"
  --warmup-steps "$WARMUP_STEPS" --lr-scheduler "$LR_SCHEDULER"
  --weight-decay "$WEIGHT_DECAY" --optimizer "$OPTIMIZER"
  --gradient-checkpointing "$GRADIENT_CHECKPOINTING"
  --gradient-checkpoint-use-reentrant "$GRADIENT_CHECKPOINT_USE_REENTRANT"
  --ddp-find-unused-parameters "$DDP_FIND_UNUSED_PARAMETERS"
  --dataloader-workers "$DATALOADER_WORKERS" --logging-first-step "$LOGGING_FIRST_STEP"
  --report-to "$REPORT_TO" --save-strategy "$SAVE_STRATEGY" --eval-strategy "$EVAL_STRATEGY"
  --save-total-limit "$SAVE_TOTAL_LIMIT" --load-best-model-at-end "$LOAD_BEST_MODEL_AT_END"
  --metric-for-best-model "$METRIC_FOR_BEST_MODEL" --greater-is-better "$GREATER_IS_BETTER"
  --remove-unused-columns "$REMOVE_UNUSED_COLUMNS" --save-safetensors "$SAVE_SAFETENSORS"
  --disable-tqdm "$DISABLE_TQDM" --eval-accumulation-steps "$EVAL_ACCUMULATION_STEPS"
  --use-cache "$USE_CACHE" --train-order "$TRAIN_ORDER"
  --max-grad-norm "$MAX_GRAD_NORM" --label-smoothing-factor "$LABEL_SMOOTHING_FACTOR"
  --dataloader-drop-last "$DATALOADER_DROP_LAST"
  --full-determinism "$FULL_DETERMINISM" --bf16-full-eval "$BF16_FULL_EVAL"
)
finish() {
  local rc=$?
  trap - EXIT
  printf '[F-01b v1] EXIT=%s MODEL_MODE=%s MODE=%s CHICAGO=%s LOG=%s RUN=%s\n' \
    "$rc" "$MODEL_MODE" "$MODE" "$(TZ=America/Chicago date -Is)" "$LOG_FILE" "$RUN_DIR" | tee -a "$LOG_FILE"
  exit "$rc"
}
trap finish EXIT
{
  printf '[F-01b v1] MODE=%s MODEL_MODE=%s MODEL_ID=%s REVISION=%s CHICAGO=%s\n' \
    "$MODE" "$MODEL_MODE" "$MODEL_ID" "$MODEL_REVISION" "$(TZ=America/Chicago date -Is)"
  printf '[F-01b v1] GPUs=%s HF_HOME=%s RUN=%s LOG=%s\n' "$GPU_IDS" "$HF_HOME" "$RUN_DIR" "$LOG_FILE"
  printf '[F-01b v1] TRAIN=%s VAL=%s STEPS=%s BATCH_GLOBAL=%s TOKEN_POLICY=%s\n' \
    "$TRAIN_EXAMPLES" "$VAL_EXAMPLES" "$MAX_STEPS" \
    "$(( PER_DEVICE_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS * EXPECTED_WORLD_SIZE ))" "$TOKEN_POLICY"
  case "$MODE" in
    --preflight)
      "$PYTHON_BIN" -u "$PYFILE" preflight "${COMMON[@]}"
      ;;
    --prepare)
      "$PYTHON_BIN" -u "$PYFILE" prepare "${COMMON[@]}"
      ;;
    --smoke|--train)
      "$PYTHON_BIN" -u "$PYFILE" prepare "${COMMON[@]}"
      "$PYTHON_BIN" -u -m torch.distributed.run --standalone \
        --nproc_per_node="$EXPECTED_WORLD_SIZE" "$PYFILE" worker "${COMMON[@]}"
      "$PYTHON_BIN" -u "$PYFILE" finalize "${COMMON[@]}"
      ;;
    --verify)
      "$PYTHON_BIN" -u "$PYFILE" verify "${COMMON[@]}"
      ;;
  esac
} 2>&1 | tee -a "$LOG_FILE"
