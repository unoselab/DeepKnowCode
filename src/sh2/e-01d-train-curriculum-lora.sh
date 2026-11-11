#!/usr/bin/env bash
# E-01d v3: all experiment settings, file paths, and logging live here.
set -Eeuo pipefail
export TZ=America/Chicago
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
IMPLEMENTATION="$ROOT/src/py2/e-01d-train-curriculum-lora.py"
ORIGINAL_D04="$ROOT/src/py2/d-04-train-curriculum-lora.py"
ORIGINAL_D04_HASH="e49298f7339f9b8299a586746cebf659288f17e83db1b157b5c4fc316d85becc"
MODEL_ID="bigcode/starcoder2-15b"
REVISION="46d44742909c03ac8cee08eb03fdebce02e193ec"
M1_DIR="$ROOT/results/e-01b/20261110-144838-92489-train/adapter"
M1_WEIGHTS_HASH="fd2c4b713a8bbc8ae8d9e0b11f3f1eb3923fbe361433628c9b691e7017b3d9ae"
M1_CONFIG_HASH="df4d929e53e062633ecb0ba8962f0193a2d3495b1ba4dee6a89d9c984e90adbd"
E01C_PREPARED="$ROOT/results/e-01c/20261110-234438-842292-train/prepared"
E01C_SUMMARY="$ROOT/results/e-01c/20261110-234438-842292-train/training_summary.json"
E01C_SUMMARY_HASH="d31659acf7c0614fbdbd4f1ba6531bc7560f0e296aebab5a418cee0938103324"
E01C_MANIFEST_HASH="75cf2c5d3026d4ecb4ebfe7f8c5a82a31bc24c874790975515f64ae1e487047b"
ORIGINAL_CANDIDATE="$ROOT/results/d-04/curriculum-audit-v5"
ORDER_HASH="005ef7e058d79d67cdd1e457b0a2657d4c41272c9762aec03d1a84da2c45ece3"
CANDIDATE_STATUS="AUDIT_ONLY_NO_PRIMARY_SCORE_SELECTED_NO_M3_TRAINING"
TRAIN_HASH="b7e2db71f36171ed9fe96bef4082b34dc80a56b3c853c852d73684e186351a1e"
VAL_HASH="331b36966e1cfeecfc4d328c9cd2acab44f5b7be7a568c1edfa6c09b6a1336d2"
TRAIN_COUNT=19464
VAL_COUNT=512
SOURCE_TRAIN=12000
JAVA_TRAIN=3732
PYTHON_TRAIN=3732
SOURCE_VAL=412
JAVA_VAL=60
PYTHON_VAL=40
BUDGET_POLICY="fixed-source"
SEED=2026
MAX_SEQ_LENGTH=2048
STEPS=811
EVAL_STEPS=811
LOG_STEPS=10
WARMUP_STEPS=20
BATCH_SIZE=1
EVAL_BATCH_SIZE=1
ACCUM=8
WORLD_SIZE_EXPECTED=3
LR="0.0002"
LORA_RANK=16
LORA_ALPHA=32
LORA_DROPOUT="0.05"
GPU_IDS="${E01D_GPUS:-0,1,2}"
AUDIT_STEPS=3
# Reuse verified v2 evidence directories unchanged; never overwrite. Audit is separate from M3 training.
PREPARED_DIR="$ROOT/results/e-01d/prepared-v2"
CANDIDATE_DIR="$ROOT/results/e-01d/curriculum-candidate-v2"
TRAINER_AUDIT_DIR="$ROOT/results/e-01d/trainer-order-audit-v2"
TRAIN_RUN_DIR="$ROOT/results/e-01d/m3-curriculum-v2"
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
# Automatic logging. Every invocation receives a unique file, no external tee needed.
LOG_DIR="$ROOT/logs/e-01d"
LOG_PREFIX="e-01d-v3"
LOG_TIMESTAMP_FORMAT="%Y-%m%d-%H%M%S"
LOG_TEE_CONSOLE=true
LOG_SAVE_ENABLED=true
MODE="${1:-}"
case "$MODE" in
  --preflight) CMD=preflight ;;
  --prepare) CMD=prepare ;;
  --audit-trainer) CMD=audit-trainer ;;
  --train) CMD=train ;;
  --verify) CMD=verify ;;
  *) echo 'Usage: bash src/sh2/e-01d-train-curriculum-lora.sh --preflight|--prepare|--audit-trainer|--train|--verify' >&2; exit 2 ;;
esac
shift
[[ $# -eq 0 ]] || { echo 'No positional overrides allowed; define settings in Shell' >&2; exit 2; }
[[ -f "$IMPLEMENTATION" ]] || { echo "Missing Python implementation $IMPLEMENTATION" >&2; exit 2; }
mkdir -p "$LOG_DIR"
STAMP="$(TZ=America/Chicago date +"$LOG_TIMESTAMP_FORMAT")"
LOG_FILE="$(mktemp "$LOG_DIR/${LOG_PREFIX}-${CMD}-${STAMP}-XXXXXX.log")"
[[ "$LOG_SAVE_ENABLED" == true && "$LOG_TEE_CONSOLE" == true ]] || { echo 'Log saving must be enabled' >&2; exit 2; }
exec > >(tee "$LOG_FILE") 2>&1
trap 'code=$?; echo "[E-01d v3] EXIT code=$code mode=$CMD chicago=$(TZ=America/Chicago date --iso-8601=seconds) log=$LOG_FILE"' EXIT
echo "[E-01d v3] Mode: $CMD"
echo "[E-01d v3] Log: $LOG_FILE"
echo "[E-01d v3] Chicago started: $(TZ=America/Chicago date --iso-8601=seconds)"
ARGS=(--root "$ROOT" --original-d04 "$ORIGINAL_D04" --original-d04-hash "$ORIGINAL_D04_HASH"
 --model-id "$MODEL_ID" --revision "$REVISION"
 --m1-dir "$M1_DIR" --m1-weights-hash "$M1_WEIGHTS_HASH" --m1-config-hash "$M1_CONFIG_HASH"
 --e01c-prepared "$E01C_PREPARED" --e01c-summary "$E01C_SUMMARY" --e01c-summary-hash "$E01C_SUMMARY_HASH" --e01c-manifest-hash "$E01C_MANIFEST_HASH"
 --original-candidate "$ORIGINAL_CANDIDATE" --candidate-status "$CANDIDATE_STATUS" --order-hash "$ORDER_HASH"
 --prepared-dir "$PREPARED_DIR" --candidate-dir "$CANDIDATE_DIR" --trainer-audit-dir "$TRAINER_AUDIT_DIR" --train-run-dir "$TRAIN_RUN_DIR"
 --train-hash "$TRAIN_HASH" --val-hash "$VAL_HASH" --train-count "$TRAIN_COUNT" --val-count "$VAL_COUNT"
 --source-train "$SOURCE_TRAIN" --java-train "$JAVA_TRAIN" --python-train "$PYTHON_TRAIN"
 --source-val "$SOURCE_VAL" --java-val "$JAVA_VAL" --python-val "$PYTHON_VAL"
 --budget-policy "$BUDGET_POLICY" --seed "$SEED" --max-seq-length "$MAX_SEQ_LENGTH"
 --steps "$STEPS" --eval-steps "$EVAL_STEPS" --log-steps "$LOG_STEPS" --warmup-steps "$WARMUP_STEPS"
 --batch-size "$BATCH_SIZE" --eval-batch-size "$EVAL_BATCH_SIZE" --accum "$ACCUM" --world-size "$WORLD_SIZE_EXPECTED"
 --lr "$LR" --lora-rank "$LORA_RANK" --lora-alpha "$LORA_ALPHA" --lora-dropout "$LORA_DROPOUT"
 --gpus "$GPU_IDS" --audit-steps "$AUDIT_STEPS" --python-bin "$PYTHON_BIN")
"$PYTHON_BIN" -u "$IMPLEMENTATION" "$CMD" "${ARGS[@]}"
