#!/usr/bin/env bash
# G-04a v1 standalone GPT-OSS-120B Python heldout evaluation; no training.
set -Eeuo pipefail
export TZ=America/Chicago
export TOKENIZERS_PARALLELISM=false
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
PYTHON_BIN="${G04A_PYTHON_BIN:-python}"
ENGINE="$ROOT/src/py2/g-04a-evaluate.py"
MODE="${1:-}"
[[ -f "$ENGINE" ]] || { echo "Missing G04A Python engine: $ENGINE" >&2; exit 2; }
case "$MODE" in
  --preflight|--evaluate)
    [[ $# -eq 2 && -d "$2" ]] || { echo "USAGE: bash $0 $MODE <G03D_verified_run_dir>" >&2; exit 2; }
    G03D_RUN="$(cd "$2" && pwd -P)"
    EVAL_RUN=""
    ;;
  --verify)
    [[ $# -eq 3 && -d "$2" && -d "$3" ]] || { echo "USAGE: bash $0 --verify <G04A_eval_run_dir> <G03D_verified_run_dir>" >&2; exit 2; }
    EVAL_RUN="$(cd "$2" && pwd -P)"
    G03D_RUN="$(cd "$3" && pwd -P)"
    ;;
  *) echo "USAGE: bash $0 --preflight|--evaluate <G03D_run>; or --verify <G04A_run> <G03D_run>" >&2; exit 2;;
esac

# Explicit Shell-side frozen experiment parameters. Python has no experiment defaults.
VERSION="v1"
MODEL_ID="openai/gpt-oss-120b"
REVISION="b5c939de8f754692c1647ca79fbf85e8c1e70f8a"
MODEL_TYPE="gpt_oss"
HF_HOME="${G04A_HF_HOME:-$HOME/.cache/huggingface}"
DEVICE_IDS="0,1,2"
EXPECTED_GPUS=3
MIN_COMPUTE_MAJOR=8
MAX_GPU_GIB=39
MAX_CPU_GIB=4
PRECISION="bf16"
QUANT_METHOD="mxfp4"
DEVICE_MAP="balanced"
ATTENTION_BACKEND="eager"
LOCAL_FILES_ONLY="true"
TRUST_REMOTE_CODE="false"
MODEL_ORDER="M0,M1,M2,M3"
M1_SHA256="0c9d913781a8a06342fef64113bfbb2ccd2d9d7dddd2b41d760915dbf8018cbf"
M2_SHA256="057a2a9e544934d4deec3ada98b05b3f7d4a3aa5e7dc2f39f444aaa7fdfcd858"
M3_SHA256="42defe62b711a331ebe0c6a27480c4547422523dcadf6e92acba09ec9d970a15"
G03B_ORDER_SHA256="9e8ad5fdaaa1729aed4d75afe5911b5f617b2b6882b7a0d338c745e0c9d54766"
G02_TRAIN_SHA256="546142986ff39e08ccf38db6b6c12284c200f5bf20775c1b0b97f1a34fb63eeb"
G02_VAL_SHA256="cf78092416d3ca8d039daf5c722490ec60d419232d17cc7a812f6a125d5b686c"
G03D_PASS_STATUS="G03D_V1_INDEPENDENT_M3_VERIFY_PASS"
G03D_MANIFEST_STATUS="G03D_V1_REPORT_MANIFEST_PASS"
EXPECTED_INIT_POLICY="independent-branches-from-frozen-M1"
TRAIN_PAIRS=19464
TRAIN_STEPS=811
TRAIN_BATCH=24
GROUND_TRUTH="$ROOT/data/evaluation/ground_truth/python"
EXPECTED_GT_SHA256="2d171cdee52f6ab8229853b0cc5590b71e1a7dd6919a417dd88fb17d0ec0cb8c"
EXPECTED_GT_MANIFEST_SHA256="309cd69ef64e4c840557fd62d3a732894cc322054f5b80d39956c1411db3cb13"
EXCLUDED_PROJECT="azure-sdk-for-python"
EXPECTED_PAIRS=7942
EXPECTED_PER_CLASS=3971
EXPECTED_PROJECTS=24
TEMPLATE_DATE="2026-11-12"
SYSTEM_TEXT="You are a code clone classifier. Return exactly YES or NO in the final answer."
USER_PREFIX=$'Determine whether the following two code fragments are code clones. Answer YES if they are clones and NO otherwise.\nLanguage: java\nCode A:\n'
USER_SEPARATOR=$'\nCode B:\n'
USER_SUFFIX=$'\nReturn YES or NO.'
ANSWER_YES=" YES"
ANSWER_NO=" NO"
PROBE_CODE_A='def foo(x): return x + 1'
PROBE_CODE_B='def bar(x): return x + 1'
MAX_INPUT_TOKENS=2048
WINDOW_OVERLAP="0.20"
MAX_WINDOWS=256
MIN_CODE_TOKENS=128
WINDOW_SAFETY_TOKENS=128
WINDOW_MAX_ITERATIONS=48
WINDOW_BACKOFF_TOKENS=32
DECISION_THRESHOLD="0.0"
SEED=2026
PROGRESS_EVERY=100
CACHE_CLEAR_EVERY=100
LOG_DIR="$ROOT/logs/g-04/gpt-oss-120b"
LOG_PREFIX="g-04a-v1"
LOG_TIMESTAMP_FORMAT='+%Y-%m%d-%H%M%S'
RESULTS_DIR="$ROOT/results/g-04/gpt-oss-120b"
mkdir -p "$LOG_DIR" "$RESULTS_DIR"
STAMP="$(TZ=America/Chicago date "$LOG_TIMESTAMP_FORMAT")"
RUN_ID="${STAMP}-$(date +%s)-$$-${RANDOM}"
if [[ "$MODE" == '--verify' ]]; then
  RUN_DIR="$EVAL_RUN"
else
  RUN_DIR="$RESULTS_DIR/$RUN_ID-${MODE#--}"
  [[ ! -e "$RUN_DIR" ]] || { echo "REFUSE_OVERWRITE_RUN:$RUN_DIR" >&2; exit 2; }
fi
LOG_FILE="$LOG_DIR/$LOG_PREFIX-${MODE#--}-$RUN_ID.log"
[[ ! -e "$LOG_FILE" ]] || { echo 'REFUSE_OVERWRITE_LOG' >&2; exit 2; }
( set -C; : > "$LOG_FILE" )
exec > >(tee -a "$LOG_FILE") 2>&1
export HF_HOME CUDA_VISIBLE_DEVICES="$DEVICE_IDS"
chicago() { TZ=America/Chicago date '+%Y-%m-%dT%H:%M:%S%:z'; }
START_EPOCH="$(date +%s)"
echo "[G-04a v1] START MODE=$MODE CHICAGO=$(chicago) RUN=$RUN_DIR LOG=$LOG_FILE G03D_RUN=$G03D_RUN"
echo "[G-04a v1] MODEL=$MODEL_ID REVISION=$REVISION MODELS=$MODEL_ORDER GT_SHA256=$EXPECTED_GT_SHA256"
echo "[G-04a v1] EVAL_N=$EXPECTED_PAIRS EACH_CLASS=$EXPECTED_PER_CLASS PROJECTS=$EXPECTED_PROJECTS THRESHOLD=$DECISION_THRESHOLD INFERENCE=$PRECISION/$QUANT_METHOD/$DEVICE_MAP"
ARGS=(
 --run-dir "$RUN_DIR" --g03d-run "$G03D_RUN" --ground-truth "$GROUND_TRUTH"
 --hf-home "$HF_HOME" --log-file "$LOG_FILE" --model-id "$MODEL_ID"
 --revision "$REVISION" --model-type "$MODEL_TYPE" --precision "$PRECISION"
 --quant-method "$QUANT_METHOD" --attention-backend "$ATTENTION_BACKEND"
 --device-map "$DEVICE_MAP" --device-ids "$DEVICE_IDS"
 --model-order "$MODEL_ORDER" --local-files-only "$LOCAL_FILES_ONLY"
 --trust-remote-code "$TRUST_REMOTE_CODE"
 --expected-gt-sha256 "$EXPECTED_GT_SHA256"
 --expected-gt-manifest-sha256 "$EXPECTED_GT_MANIFEST_SHA256"
 --excluded-project "$EXCLUDED_PROJECT"
 --m1-sha256 "$M1_SHA256" --m2-sha256 "$M2_SHA256" --m3-sha256 "$M3_SHA256"
 --g03b-order-sha256 "$G03B_ORDER_SHA256" --g02-train-sha256 "$G02_TRAIN_SHA256"
 --g02-validation-sha256 "$G02_VAL_SHA256" --g03d-pass-status "$G03D_PASS_STATUS"
 --g03d-manifest-status "$G03D_MANIFEST_STATUS" --expected-init-policy "$EXPECTED_INIT_POLICY"
 --train-pairs "$TRAIN_PAIRS" --train-steps "$TRAIN_STEPS" --train-batch "$TRAIN_BATCH"
 --expected-pairs "$EXPECTED_PAIRS" --expected-per-class "$EXPECTED_PER_CLASS"
 --expected-projects "$EXPECTED_PROJECTS" --max-input-tokens "$MAX_INPUT_TOKENS"
 --max-windows "$MAX_WINDOWS" --min-code-tokens "$MIN_CODE_TOKENS"
 --window-safety-tokens "$WINDOW_SAFETY_TOKENS"
 --window-max-iterations "$WINDOW_MAX_ITERATIONS"
 --window-backoff-tokens "$WINDOW_BACKOFF_TOKENS"
 --progress-every "$PROGRESS_EVERY" --cache-clear-every "$CACHE_CLEAR_EVERY"
 --seed "$SEED" --expected-gpus "$EXPECTED_GPUS" --min-compute-major "$MIN_COMPUTE_MAJOR"
 --max-gpu-gib "$MAX_GPU_GIB" --max-cpu-gib "$MAX_CPU_GIB"
 --window-overlap "$WINDOW_OVERLAP" --threshold "$DECISION_THRESHOLD"
 --template-date "$TEMPLATE_DATE" --system-text "$SYSTEM_TEXT"
 --user-prefix "$USER_PREFIX" --user-separator "$USER_SEPARATOR" --user-suffix "$USER_SUFFIX"
 --answer-yes "$ANSWER_YES" --answer-no "$ANSWER_NO"
 --probe-code-a "$PROBE_CODE_A" --probe-code-b "$PROBE_CODE_B"
)

stage() {
  local step="$1"; shift
  local started rc
  started="$(date +%s)"
  echo "[G-04a v1] STEP=$step START CHICAGO=$(chicago)"
  if "$PYTHON_BIN" -u "$ENGINE" "$@" "${ARGS[@]}"; then
    echo "[G-04a v1] STEP=$step PASS EXIT=0 ELAPSED_SECONDS=$(($(date +%s)-started)) CHICAGO=$(chicago)"
  else
    rc=$?
    echo "[G-04a v1] STEP=$step FAIL EXIT=$rc ELAPSED_SECONDS=$(($(date +%s)-started)) CHICAGO=$(chicago)"
    echo "[G-04a v1] FINAL_STATUS=FAIL EXIT=$rc TOTAL_SECONDS=$(($(date +%s)-START_EPOCH)) RUN=$RUN_DIR LOG=$LOG_FILE"
    exit "$rc"
  fi
}
case "$MODE" in
  --preflight) stage PREFLIGHT preflight;;
  --evaluate)
    stage PREFLIGHT preflight
    IFS=',' read -r -a MODELS <<< "$MODEL_ORDER"
    for model in "${MODELS[@]}"; do
      stage "INFER_$model" infer --model "$model"
    done
    stage SUMMARIZE summarize
    stage VERIFY verify
    ;;
  --verify) stage VERIFY verify;;
esac
echo "[G-04a v1] FINAL_STATUS=PASS EXIT=0 TOTAL_SECONDS=$(($(date +%s)-START_EPOCH)) RUN=$RUN_DIR LOG=$LOG_FILE"
