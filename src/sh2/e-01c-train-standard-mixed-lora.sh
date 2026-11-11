#!/usr/bin/env bash
# E-01c v2: ALL experiment parameters and paths are defined here.
set -Eeuo pipefail
export TZ=America/Chicago
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
IMPLEMENTATION="$ROOT/src/py2/e-01c-train-standard-mixed-lora.py"
ORIGINAL_M2="$ROOT/src/py2/d-03a-train-balanced-mixed-lora.py"
MODEL_ID="bigcode/starcoder2-15b"
REVISION="46d44742909c03ac8cee08eb03fdebce02e193ec"
M1_RUN_DIR="$ROOT/results/e-01b/20261110-144838-92489-train"
M1_EXPECTED_STEPS=500
M1_PREPARED_SHA256="c30c94db158210e5c2e4854d7a7e605c50720d250f37fcb96c03ed4e9792f8da"
M1_CONFIG_SHA256="df4d929e53e062633ecb0ba8962f0193a2d3495b1ba4dee6a89d9c984e90adbd"
M1_WEIGHTS_SHA256="fd2c4b713a8bbc8ae8d9e0b11f3f1eb3923fbe361433628c9b691e7017b3d9ae"
M1_SOURCE_SHA256="688135d21eb6b3ec132a2ca4c35f00a0bfeba389d2c63c5265856d2d810fb2bc"
GROUND_TRUTH_SHA256="309cd69ef64e4c840557fd62d3a732894cc322054f5b80d39956c1411db3cb13"
FROZEN_DIR="$ROOT/data/training/prepared/balanced_multilingual"
FROZEN_MODEL_ID="bigcode/starcoder2-7b"
FROZEN_REVISION="bb9afde76d7945da5745592525db122d4d729eb1"
FROZEN_TRAIN_SHA256="b7e2db71f36171ed9fe96bef4082b34dc80a56b3c853c852d73684e186351a1e"
FROZEN_VALIDATION_SHA256="331b36966e1cfeecfc4d328c9cd2acab44f5b7be7a568c1edfa6c09b6a1336d2"
FROZEN_TRAIN_EXAMPLES=19464
FROZEN_VALIDATION_EXAMPLES=512
SOURCE_TRAIN=12000
JAVA_TARGET_TRAIN=3732
PYTHON_TARGET_TRAIN=3732
SOURCE_VAL=412
JAVA_TARGET_VAL=60
PYTHON_TARGET_VAL=40
BUDGET_POLICY="fixed-source"
SEED=2026
MAX_SEQ_LENGTH=2048
EVAL_STEPS=811
LOG_STEPS=10
WARMUP_STEPS=20
PER_DEVICE_BATCH_SIZE=1
PER_DEVICE_EVAL_BATCH_SIZE=1
GRAD_ACCUM_STEPS=8
EXPECTED_WORLD_SIZE=3
LEARNING_RATE="0.0002"
LORA_RANK=16
LORA_ALPHA=32
LORA_DROPOUT="0.05"
GPU_IDS="${E01C_GPUS:-0,1,2}"
EXPECTED_ANSWER_NO=4452
EXPECTED_ANSWER_YES=24043
PYTHON_MIXED_DATA_SHA256="da8fdd6362df692e225a5bf5b72278fd6211cef75e48e6a539dda637b76a1843"
PYTHON_MIXED_TRAIN_SHA256="e8c8998f2e73a3587190904fb258ac1ccda306f0be62dec499622f4258603693"
PYTHON_MIXED_VALID_SHA256="33c5a0506fdbbd8ff5f9604a9be1d4fba1b6b6d52943ff11ff53928861ac48a7"
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
[[ $# -ge 1 ]] || { echo 'Usage: --preflight|--prepare|--train|--verify [--run-dir DIR]' >&2; exit 2; }
MODE="$1"; shift
case "$MODE" in --preflight) CMD=preflight;; --prepare) CMD=prepare;; --train) CMD=train;; --verify) CMD=verify;; *) echo "Invalid mode: $MODE" >&2; exit 2;; esac
[[ -f "$IMPLEMENTATION" ]] || { echo "Missing $IMPLEMENTATION" >&2; exit 2; }
args=("$CMD" --root "$ROOT" --original-m2 "$ORIGINAL_M2" --model-id "$MODEL_ID" --revision "$REVISION"
 --m1-run-dir "$M1_RUN_DIR" --m1-expected-steps "$M1_EXPECTED_STEPS" --m1-prepared-sha256 "$M1_PREPARED_SHA256" --m1-config-sha256 "$M1_CONFIG_SHA256"
 --m1-weights-sha256 "$M1_WEIGHTS_SHA256" --m1-source-sha256 "$M1_SOURCE_SHA256" --ground-truth-sha256 "$GROUND_TRUTH_SHA256"
 --frozen-dir "$FROZEN_DIR" --frozen-model-id "$FROZEN_MODEL_ID" --frozen-revision "$FROZEN_REVISION"
 --frozen-train-sha256 "$FROZEN_TRAIN_SHA256" --frozen-validation-sha256 "$FROZEN_VALIDATION_SHA256"
 --frozen-train-examples "$FROZEN_TRAIN_EXAMPLES" --frozen-validation-examples "$FROZEN_VALIDATION_EXAMPLES"
 --source-train "$SOURCE_TRAIN" --java-target-train "$JAVA_TARGET_TRAIN" --python-target-train "$PYTHON_TARGET_TRAIN"
 --source-val "$SOURCE_VAL" --java-target-val "$JAVA_TARGET_VAL" --python-target-val "$PYTHON_TARGET_VAL"
 --budget-policy "$BUDGET_POLICY" --seed "$SEED" --max-seq-length "$MAX_SEQ_LENGTH" --eval-steps "$EVAL_STEPS"
 --log-steps "$LOG_STEPS" --warmup-steps "$WARMUP_STEPS" --per-device-batch-size "$PER_DEVICE_BATCH_SIZE"
 --per-device-eval-batch-size "$PER_DEVICE_EVAL_BATCH_SIZE" --gradient-accumulation-steps "$GRAD_ACCUM_STEPS"
 --expected-world-size "$EXPECTED_WORLD_SIZE" --learning-rate "$LEARNING_RATE" --lora-rank "$LORA_RANK"
 --lora-alpha "$LORA_ALPHA" --lora-dropout "$LORA_DROPOUT" --gpus "$GPU_IDS" --answer-no "$EXPECTED_ANSWER_NO"
 --answer-yes "$EXPECTED_ANSWER_YES" --python-mixed-data-sha256 "$PYTHON_MIXED_DATA_SHA256"
 --python-mixed-train-sha256 "$PYTHON_MIXED_TRAIN_SHA256" --python-mixed-valid-sha256 "$PYTHON_MIXED_VALID_SHA256")
exec "$PYTHON_BIN" -u "$IMPLEMENTATION" "${args[@]}" "$@"
