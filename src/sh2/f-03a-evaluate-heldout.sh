#!/usr/bin/env bash
# F-03a v1: frozen M0/M1/M2/M3 CodeLlama Python held-out evaluation.
# Shell owns every experiment parameter. Never invokes another experiment shell.
set -Eeuo pipefail
export TZ=America/Chicago
export TOKENIZERS_PARALLELISM=false
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
PYTHON_BIN="${F03A_PYTHON_BIN:-python}"
PYFILE="$ROOT/src/py2/f-03a-evaluate-heldout.py"
MODE="${1:---evaluate}"
if (( $# )); then shift; fi
[[ $# -eq 0 ]] || { echo "Usage: bash $0 --preflight|--evaluate" >&2;exit 2; }
case "$MODE" in --preflight|--evaluate) ;; *) echo "Unsupported mode: $MODE" >&2; exit 2;; esac
[[ -f "$PYFILE" ]] || { echo "Missing Python implementation: $PYFILE" >&2;exit 2; }

# All experiment and audit controls, explicitly selected in this wrapper.
EXPERIMENT="F-03a"
VERSION="v1"
MODEL_MODE="codellama-7b"
MODEL_ID="meta-llama/CodeLlama-7b-hf"
EXPECTED_MODEL_TYPE="llama"
REVISION="b462c3c99b077d341db691ec780a33156f3c1472"
HF_HOME="${F03A_HF_HOME:-$HOME/.cache/huggingface}"
GPU_ID="${F03A_GPU_ID:-0}"
LOAD_MODE="4bit"  # E-01e/D-05c inference uses NF4; same policy across M0-M3.
LOCAL_FILES_ONLY="true"
TRUST_REMOTE_CODE="false"
EXPECTED_PAIRS=7942
EXPECTED_PER_CLASS=3971
EXPECTED_PROJECTS=24
EXCLUDED_PROJECT="azure-sdk-for-python"
EXPECTED_GT_SHA256="2d171cdee52f6ab8229853b0cc5590b71e1a7dd6919a417dd88fb17d0ec0cb8c"
GROUND_TRUTH="$ROOT/data/evaluation/ground_truth/python"
MAX_INPUT_TOKENS=4096
WINDOW_OVERLAP="0.20"
MAX_WINDOWS=256
PROGRESS_EVERY=100
SEED=2026
ANSWER_YES=" YES"
ANSWER_NO=" NO"
YES_TOKEN_ID=22483
NO_TOKEN_ID=11698
DECISION_THRESHOLD="0"
PROBE_CODE_A='def foo(x): return x + 1'
PROBE_CODE_B='def bar(x): return x + 1'
EXPECTED_TRAIN_STEPS=811
FROZEN_PREPARED="$ROOT/results/f-01c/codellama-7b/2026-1112-115525-prepare/prepared"
FROZEN_TRAIN="$FROZEN_PREPARED/train.jsonl"
FROZEN_VALIDATION="$FROZEN_PREPARED/validation.jsonl"
FROZEN_TRAIN_SHA256="e2ec9d2cccc0adb0b73f81f60b3041b5a7f4fa519729f501c1cdc583c1e9b564"
FROZEN_VALIDATION_SHA256="4404f90262c1021917ce31f0c454f87a5a634a0875307b8e84bbd51abefc752a"
CURRICULUM_RUN="$ROOT/results/f-01e/codellama-7b/2026-1112-155705-1794520625-239724-11041-prepare"
CURRICULUM_ORDER="$CURRICULUM_RUN/curriculum_order.jsonl"
CURRICULUM_ORDER_SHA256="ca0ae183c3c8b4c86f2db091c8faf18f66c545e8f6b18335c4b1d361af881566"
M1_RUN="$ROOT/results/f-01b/codellama-7b/2026-1111-232648-train"
M1_ADAPTER="$M1_RUN/adapter"
M1_TRAINING_SUMMARY="$M1_RUN/training_summary.json"
M1_ADAPTER_SHA256="b8f057ca2e413402738ab7484c0b0108346f4d1c991f00e80ec21508514c78b4"
M3_RUN="$ROOT/results/f-02a/codellama-7b/2026-1112-182726-1794529646-245291-23299-train"
M3_ADAPTER="$M3_RUN/adapter"
M3_TRAINING_SUMMARY="$M3_RUN/full_training_summary.json"
M3_TRAINING_EVIDENCE="$M3_RUN/full_training_evidence.json"
M3_TRAINING_EVIDENCE_SHA256="dd2fa3886b9731095de571fda692a66a52efe95360d908bbd996f795471be422"
M3_ADAPTER_SHA256="374d95cbaffb3645c47e051412b0871c296d79cfcdf1ab339befd7261943734c"

# M2 run name is not user-supplied; selection is explicitly controlled and fails
# closed on ambiguity. Override with F03A_M2_TRAIN_RUN when multiple runs exist.
M2_RUN_SELECTION="unique-approved-f01c-v24-or-explicit"
M2_RUN="${F03A_M2_TRAIN_RUN:-}"
if [[ -z "$M2_RUN" ]]; then
  shopt -s nullglob
  candidates=( "$ROOT"/results/f-01c/codellama-7b/*-train/f01c_m2_manifest.json )
  shopt -u nullglob
  [[ ${#candidates[@]} -eq 1 ]] || {
    echo "BLOCKED: M2 run ambiguous/missing (${#candidates[@]} candidates). Set F03A_M2_TRAIN_RUN to the completed M2 -train run." >&2
    printf '%s\n' "${candidates[@]:-}" >&2
    exit 2
  }
  M2_RUN="$(dirname "${candidates[0]}")"
fi
[[ -d "$M2_RUN" && -f "$M2_RUN/f01c_m2_manifest.json" ]] || { echo "BLOCKED: invalid approved M2 run: $M2_RUN" >&2; exit 2; }
M2_RUN="$(cd "$M2_RUN" && pwd -P)"
M2_ADAPTER="$M2_RUN/adapter"
M2_TRAINING_SUMMARY="$M2_RUN/training_summary.json"
M2_APPROVAL_MANIFEST="$M2_RUN/f01c_m2_manifest.json"
[[ -f "$M2_TRAINING_SUMMARY" && -f "$M2_ADAPTER/adapter_model.safetensors" ]] || { echo 'BLOCKED: M2 summary/weight missing' >&2; exit 2; }
# Derive the explicitly bound M2 weight pin from its F-01c completed training
# summary, not from the weight itself or a record-only bypass.
M2_ADAPTER_SHA256="$($PYTHON_BIN - "$M2_TRAINING_SUMMARY" "$EXPECTED_TRAIN_STEPS" <<'PY'
import json,sys,re
x=json.load(open(sys.argv[1],encoding='utf-8'))
assert x.get('status')=='TRAINING_COMPLETE' and x.get('global_step')==int(sys.argv[2]),'M2 incomplete'
h=x['adapter_sha256']['adapter_model.safetensors']
assert re.fullmatch('[0-9a-f]{64}',h),'Bad M2 adapter pin'
print(h)
PY
)"

# Independent single-run and automatic log identity; never overwrite.
LOG_DIR="$ROOT/logs/f-03a/$MODEL_MODE"
LOG_PREFIX="f-03a-v1"
LOG_TIMESTAMP_FORMAT='+%Y-%m%d-%H%M%S'
RESULTS_DIR="$ROOT/results/f-03a/$MODEL_MODE"
mkdir -p "$LOG_DIR" "$RESULTS_DIR"
STAMP="$(TZ=America/Chicago date "$LOG_TIMESTAMP_FORMAT")"
RUN_ID="${STAMP}-$(date +%s)-$$-${RANDOM}"
RUN_DIR="$RESULTS_DIR/$RUN_ID-${MODE#--}"
LOG_FILE="$LOG_DIR/$LOG_PREFIX-${MODE#--}-$RUN_ID.log"
[[ ! -e "$RUN_DIR" ]] || { echo "Refusing to overwrite run $RUN_DIR" >&2;exit 2; }
( set -C; : > "$LOG_FILE" ) || { echo "Refusing to overwrite log $LOG_FILE" >&2; exit 2; }
export CUDA_VISIBLE_DEVICES="$GPU_ID"
export HF_HOME
export OMP_NUM_THREADS="${F03A_OMP_NUM_THREADS:-4}"

ARGS=(
 --root "$ROOT" --run-dir "$RUN_DIR" --ground-truth "$GROUND_TRUTH" --hf-home "$HF_HOME"
 --adapter-m1 "$M1_ADAPTER" --adapter-m2 "$M2_ADAPTER" --adapter-m3 "$M3_ADAPTER"
 --m1-training-summary "$M1_TRAINING_SUMMARY" --m2-training-summary "$M2_TRAINING_SUMMARY"
 --m3-training-summary "$M3_TRAINING_SUMMARY"
 --m2-approval-manifest "$M2_APPROVAL_MANIFEST" --m3-training-evidence "$M3_TRAINING_EVIDENCE"
 --frozen-train "$FROZEN_TRAIN" --frozen-validation "$FROZEN_VALIDATION"
 --curriculum-order "$CURRICULUM_ORDER"
 --model-id "$MODEL_ID" --revision "$REVISION" --expected-model-type "$EXPECTED_MODEL_TYPE" --expected-gt-sha256 "$EXPECTED_GT_SHA256"
 --excluded-project "$EXCLUDED_PROJECT"
 --adapter-m1-sha256 "$M1_ADAPTER_SHA256" --adapter-m2-sha256 "$M2_ADAPTER_SHA256"
 --adapter-m3-sha256 "$M3_ADAPTER_SHA256" --m3-training-evidence-sha256 "$M3_TRAINING_EVIDENCE_SHA256" --curriculum-order-sha256 "$CURRICULUM_ORDER_SHA256"
 --frozen-train-sha256 "$FROZEN_TRAIN_SHA256" --frozen-validation-sha256 "$FROZEN_VALIDATION_SHA256"
 --load-mode "$LOAD_MODE" --answer-yes "$ANSWER_YES" --answer-no "$ANSWER_NO"
 --probe-code-a "$PROBE_CODE_A" --probe-code-b "$PROBE_CODE_B"
 --expected-pairs "$EXPECTED_PAIRS" --expected-per-class "$EXPECTED_PER_CLASS"
 --expected-projects "$EXPECTED_PROJECTS" --max-input-tokens "$MAX_INPUT_TOKENS"
 --max-windows "$MAX_WINDOWS" --progress-every "$PROGRESS_EVERY" --seed "$SEED"
 --gpu-id "$GPU_ID" --yes-token-id "$YES_TOKEN_ID" --no-token-id "$NO_TOKEN_ID"
 --expected-train-steps "$EXPECTED_TRAIN_STEPS" --overlap "$WINDOW_OVERLAP"
 --threshold "$DECISION_THRESHOLD" --local-files-only "$LOCAL_FILES_ONLY"
 --trust-remote-code "$TRUST_REMOTE_CODE"
)
step() {
  local name="$1";shift
  local start=$SECONDS rc=0
  echo "[F-03a v1] STEP=$name START CHICAGO=$(TZ=America/Chicago date -Is)"
  "$@" || rc=$?
  if (( rc )); then
    echo "[F-03a v1] STEP=$name FAIL EXIT=$rc ELAPSED_SECONDS=$((SECONDS-start)) CHICAGO=$(TZ=America/Chicago date -Is)"
    return "$rc"
  fi
  echo "[F-03a v1] STEP=$name PASS ELAPSED_SECONDS=$((SECONDS-start)) CHICAGO=$(TZ=America/Chicago date -Is)"
}
run_all() {
  local rc=0 start=$SECONDS
  echo "[F-03a v1] START MODE=$MODE CHICAGO=$(TZ=America/Chicago date -Is) RUN=$RUN_DIR LOG=$LOG_FILE"
  echo "[F-03a v1] CONFIG model=$MODEL_ID revision=$REVISION gpu=$GPU_ID load=$LOAD_MODE pairs=$EXPECTED_PAIRS seed=$SEED"
  echo "[F-03a v1] M2_SELECTION=$M2_RUN_SELECTION M2_RUN=$M2_RUN"
  echo "[F-03a v1] MODEL_PINS M1=$M1_ADAPTER_SHA256 M2=$M2_ADAPTER_SHA256 M3=$M3_ADAPTER_SHA256"
  step "PREFLIGHT_FROZEN_DATA_AND_ADAPTER_PINS" "$PYTHON_BIN" -u "$PYFILE" preflight "${ARGS[@]}" || rc=$?
  if (( rc==0 )) && [[ "$MODE" == --evaluate ]]; then
    step "VERIFY_EVALUATION_DEPENDENCIES" "$PYTHON_BIN" -c 'import torch,transformers,peft,sklearn,bitsandbytes; print("F03A dependencies PASS")' || rc=$?
    if (( rc==0 )); then
      for mid in M0 M1 M2 M3; do
        step "INFER_$mid" "$PYTHON_BIN" -u "$PYFILE" infer "${ARGS[@]}" --model "$mid" || { rc=$?; break; }
      done
    fi
    if (( rc==0 )); then step "SUMMARY_AND_MANIFEST" "$PYTHON_BIN" -u "$PYFILE" summarize "${ARGS[@]}" || rc=$?; fi
  fi
  local status=PASS
  if (( rc!=0 )); then status=FAIL; fi
  echo "[F-03a v1] FINAL_STATUS=$status EXIT=$rc ELAPSED_SECONDS=$((SECONDS-start)) CHICAGO=$(TZ=America/Chicago date -Is) RUN=$RUN_DIR LOG=$LOG_FILE"
  return "$rc"
}
run_all 2>&1 | tee -a "$LOG_FILE"
