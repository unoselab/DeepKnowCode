#!/usr/bin/env bash
# E-01e v3 independent multi-mode StarCoder2 7B/15B evaluation. Never calls an existing wrapper.
set -Eeuo pipefail
export TZ=America/Chicago
export TOKENIZERS_PARALLELISM=false

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN="python"
PYFILE="$ROOT/src/py2/e-01e-evaluate-python-binary.py"
MODEL_MODE="starcoder2-15b"  # Alternative: "starcoder2-7b"; extensible to other LLM families.

# Frozen D-05c evaluation protocol; shared across 7B and 15B.
GROUND_TRUTH_DIR="$ROOT/data/evaluation/ground_truth/python"
# The D-05bj verified-pairs hash is independently checked by its signed manifest.
# record-only: record true computed hash, rather than assume unrelated 309cd... hash is GT.
EXPECTED_GT_SHA256="record-only"
EXPECTED_PAIRS=7942
EXPECTED_PER_CLASS=3971
EXPECTED_PROJECTS=24
LOAD_MODE="4bit"
MAX_INPUT_TOKENS=4096
WINDOW_OVERLAP="0.20"
MAX_WINDOWS=256
PROGRESS_EVERY=100
SEED=2026
GPU_ID=0
LOCAL_FILES_ONLY=1
TRUST_REMOTE_CODE=0
ANSWER_YES=" YES"
ANSWER_NO=" NO"
DECISION_THRESHOLD="0"

# 7B D-05c pinned model and frozen registry.
MODEL_7B="bigcode/starcoder2-7b"
REVISION_7B="bb9afde76d7945da5745592525db122d4d729eb1"
REGISTRY_7B="$ROOT/models/registry/benchmark-20261108-v1"

# 15B E-01b / E-01c / E-01d final adapters.
MODEL_15B="bigcode/starcoder2-15b"
REVISION_15B="46d44742909c03ac8cee08eb03fdebce02e193ec"
ADAPTER_M1="$ROOT/results/e-01b/20261110-144838-92489-train/adapter"
ADAPTER_M2="$ROOT/results/e-01c/20261110-234438-842292-train/adapter"
ADAPTER_M3="$ROOT/results/e-01d/m3-curriculum-v2/adapter"
ADAPTER_M1_SHA256="fd2c4b713a8bbc8ae8d9e0b11f3f1eb3923fbe361433628c9b691e7017b3d9ae"
ADAPTER_M2_SHA256="d02c251a19c6c66eb3ae93462552c2bc78e516f7beb7f8b69897274d99bf6e4a"
ADAPTER_M3_SHA256="213f6acb007aef5a885125a0459e3a6a550e6f26eb84726940309df2ef47d8d7"
ADAPTER_M1_CONFIG_SHA256="df4d929e53e062633ecb0ba8962f0193a2d3495b1ba4dee6a89d9c984e90adbd"
# M2 config hash was not independently supplied; record and validate model family/LoRA.
ADAPTER_M2_CONFIG_SHA256="record-only"
ADAPTER_M3_CONFIG_SHA256="6f21a7b69611d662a5d4586e72258eee02a6a8fe7ab6469d5db34b8b5f95c8e6"

if [[ "$MODEL_MODE" == "starcoder2-7b" ]]; then
  MODEL_ID="$MODEL_7B"
  REVISION="$REVISION_7B"
elif [[ "$MODEL_MODE" == "starcoder2-15b" ]]; then
  MODEL_ID="$MODEL_15B"
  REVISION="$REVISION_15B"
else
  echo "ERROR: Unsupported MODEL_MODE. Supported: starcoder2-7b, starcoder2-15b" >&2; exit 2
fi

RUN_DIR="$ROOT/results/e-01e/${MODEL_MODE}-v2"
LOG_DIR="$ROOT/logs/e-01e"
LOG_PREFIX="e-01e-v3-${MODEL_MODE}"
LOG_TIMESTAMP_FORMAT="%Y-%m%d-%H%M%S"
LOG_SAVE_ENABLED=true
LOG_TEE_CONSOLE=true

MODE="${1:---preflight}"
if (($#)); then shift; fi
case "$MODE" in
  --preflight|--evaluate|--summarize) [[ $# -eq 0 ]] || { echo "No extra args allowed" >&2; exit 2; } ;;
  --infer) [[ $# -eq 1 && "$1" =~ ^M[0-3]$ ]] || { echo 'Usage: --infer M0|M1|M2|M3' >&2; exit 2; } ;;
  *) echo 'Usage: --preflight|--evaluate|--infer M0|M1|M2|M3|--summarize' >&2;exit 2;;
esac
[[ -f "$PYFILE" ]] || { echo "ERROR: Missing Python implementation $PYFILE" >&2;exit 2; }
[[ "$GPU_ID" =~ ^[0-9]+$ ]] || { echo 'ERROR invalid GPU_ID' >&2;exit 2; }
export CUDA_VISIBLE_DEVICES="$GPU_ID"
mkdir -p "$LOG_DIR"
STAMP="$(TZ=America/Chicago date "+$LOG_TIMESTAMP_FORMAT")"
LOG_FILE="$LOG_DIR/${LOG_PREFIX}-${MODE#--}-${STAMP}.log"
# Atomic no-clobber creation: same-second collision is explicit error (never add suffix).
( set -C; : > "$LOG_FILE" ) || { echo "ERROR log exists: $LOG_FILE" >&2;exit 2; }
trap 'ret=$?; echo "[E-01e v3] EXIT=$ret MODE='$MODE' MODEL_MODE='$MODEL_MODE' LOG='$LOG_FILE' CHICAGO=$(TZ=America/Chicago date --iso-8601=seconds)"' EXIT
exec > >(tee -a "$LOG_FILE") 2>&1
printf '[E-01e v3] MODE=%s MODEL_MODE=%s RUN=%s LOG=%s CHICAGO=%s\n' \
  "$MODE" "$MODEL_MODE" "$RUN_DIR" "$LOG_FILE" "$(TZ=America/Chicago date --iso-8601=seconds)"

ARGS=(--model-mode "$MODEL_MODE" --model-id "$MODEL_ID" --revision "$REVISION"
 --root "$ROOT" --ground-truth "$GROUND_TRUTH_DIR" --run-dir "$RUN_DIR"
 --registry-7b "$REGISTRY_7B"
 --adapter-m1 "$ADAPTER_M1" --adapter-m2 "$ADAPTER_M2" --adapter-m3 "$ADAPTER_M3"
 --adapter-m1-sha256 "$ADAPTER_M1_SHA256" --adapter-m2-sha256 "$ADAPTER_M2_SHA256" --adapter-m3-sha256 "$ADAPTER_M3_SHA256"
 --adapter-m1-config-sha256 "$ADAPTER_M1_CONFIG_SHA256"
 --adapter-m2-config-sha256 "$ADAPTER_M2_CONFIG_SHA256"
 --adapter-m3-config-sha256 "$ADAPTER_M3_CONFIG_SHA256"
 --expected-gt-sha256 "$EXPECTED_GT_SHA256" --expected-pairs "$EXPECTED_PAIRS"
 --expected-per-class "$EXPECTED_PER_CLASS" --expected-projects "$EXPECTED_PROJECTS"
 --load-mode "$LOAD_MODE" --max-input-tokens "$MAX_INPUT_TOKENS"
 --window-overlap "$WINDOW_OVERLAP" --max-windows "$MAX_WINDOWS" --progress-every "$PROGRESS_EVERY"
 --seed "$SEED" --gpu-id "$GPU_ID" --local-files-only "$LOCAL_FILES_ONLY"
 --trust-remote-code "$TRUST_REMOTE_CODE" --answer-yes "$ANSWER_YES" --answer-no "$ANSWER_NO"
 --decision-threshold "$DECISION_THRESHOLD")

if [[ "$MODE" == '--preflight' ]]; then
  "$PYTHON_BIN" -u "$PYFILE" preflight "${ARGS[@]}"
  echo '[E-01e v3] PREFLIGHT_ONLY; no GPU inference'
elif [[ "$MODE" == '--infer' ]]; then
  "$PYTHON_BIN" -u "$PYFILE" infer "${ARGS[@]}" --model "$1"
elif [[ "$MODE" == '--summarize' ]]; then
  "$PYTHON_BIN" -u "$PYFILE" summarize "${ARGS[@]}"
else
  # --evaluate requires a prior successful --preflight; never deletes or replaces it.
  [[ -f "$RUN_DIR/preflight.json" ]] || { echo 'ERROR: run --preflight first' >&2; exit 2; }
  "$PYTHON_BIN" -c 'import torch,transformers,peft,sklearn; print("Evaluation deps OK")'
  if [[ "$LOAD_MODE" == "4bit" ]]; then "$PYTHON_BIN" -c 'import bitsandbytes; print("4bit dependency OK")'; fi
  for model in M0 M1 M2 M3; do
    echo "[E-01e v3] START $model mode=$MODEL_MODE"
    "$PYTHON_BIN" -u "$PYFILE" infer "${ARGS[@]}" --model "$model"
  done
  "$PYTHON_BIN" -u "$PYFILE" summarize "${ARGS[@]}"
  echo '[E-01e v3] ALL_MODELS_COMPLETE'
fi
