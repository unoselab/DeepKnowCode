#!/usr/bin/env bash
set -Eeuo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd)"
PY_SCRIPT="$ROOT/src/py2/d-01-controlled-python-lora.py"
if [[ -n "${D01_PYTHON:-}" ]]; then
  PYTHON="$D01_PYTHON"
elif [[ -x "$HOME/miniconda3/envs/deepknowcode/bin/python" ]]; then
  PYTHON="$HOME/miniconda3/envs/deepknowcode/bin/python"
else
  PYTHON="$(command -v python3)"
fi
[[ -f "$PY_SCRIPT" ]] || { echo "D-01 missing: $PY_SCRIPT" >&2; exit 2; }
MODE="${1:---dry-run}"
shift || true
# Wrapper-specific option; all other flags are forwarded to the Python implementation.
EXPLICIT_RUN_DIR=""
EXTRA_ARGS=()
while (($#)); do
  case "$1" in
    --run-dir)
      (($# >= 2)) || { echo 'D-01 --run-dir requires a directory' >&2; exit 2; }
      EXPLICIT_RUN_DIR="$2"; shift 2 ;;
    --run-dir=*)
      EXPLICIT_RUN_DIR="${1#--run-dir=}"; shift ;;
    *) EXTRA_ARGS+=("$1"); shift ;;
  esac
done
set -- "${EXTRA_ARGS[@]}"
case "$MODE" in
  --dry-run|--prepare|--train) ;;
  *) echo "Usage: bash src/sh2/d-01-controlled-python-lora.sh [--dry-run|--prepare|--train] [extra Python flags]" >&2; exit 2 ;;
esac
export TZ=America/Chicago
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
DATASET="${D01_DATASET:-data/training/clean/python_mixed}"
COMMON=(--root "$ROOT" --dataset "$DATASET" --model-id "${D01_MODEL_ID:-bigcode/starcoder2-7b}" \
    --revision "${D01_REVISION:-bb9afde76d7945da5745592525db122d4d729eb1}" \
    --max-seq-length "${D01_MAX_SEQ_LENGTH:-2048}" --max-steps "${D01_MAX_STEPS:-500}" \
    --expected-world-size "${D01_WORLD_SIZE:-3}" --batch-size "${D01_BATCH_SIZE:-1}" \
    --grad-accum "${D01_GRAD_ACCUM:-8}" --lr "${D01_LR:-0.0002}")
if [[ -n "${D01_APPROVAL:-}" ]]; then COMMON+=(--approval "$D01_APPROVAL"); fi
if [[ -n "${D01_INITIAL_ADAPTER:-}" ]]; then COMMON+=(--initial-adapter "$D01_INITIAL_ADAPTER"); fi
if [[ "${D01_ALLOW_DOWNLOAD:-0}" == 1 ]]; then COMMON+=(--allow-download); fi
if [[ "$MODE" == --dry-run ]]; then
  exec "$PYTHON" "$PY_SCRIPT" audit "${COMMON[@]}" "$@"
fi
mkdir -p "$ROOT/logs/d-01" "$ROOT/results/d-01"
if [[ "$MODE" == --prepare ]]; then
  [[ -z "$EXPLICIT_RUN_DIR" ]] || { echo 'D-01 --prepare creates a new run automatically; --run-dir is train-only' >&2; exit 2; }
  STAMP="$(date +%Y%m%d-%H%M%S)-$$"
  RUN="$ROOT/results/d-01/$STAMP"
  mkdir -- "$RUN"
  LOG="$ROOT/logs/d-01/$STAMP.log"
  echo "D-01 RUN=$RUN"
  echo "D-01 LOG=$LOG"
  "$PYTHON" "$PY_SCRIPT" prepare "${COMMON[@]}" --run-dir "$RUN" "$@" 2>&1 | tee "$LOG"
  echo "D-01 PREPARED RUN=$RUN"
  exit 0
fi
# Training defaults: automatically reuse the most recent valid --prepare run.
# Priority: --run-dir CLI > D01_RUN_DIR optional environment > latest prepared run.
if [[ -n "$EXPLICIT_RUN_DIR" ]]; then
  SELECTED_RUN="$EXPLICIT_RUN_DIR"
elif [[ -n "${D01_RUN_DIR:-}" ]]; then
  SELECTED_RUN="$D01_RUN_DIR"
else
  SELECTED_RUN=""
  # Run IDs are Chicago local-time YYYYMMDD-HHMMSS-pid, so lexical order is chronological.
  shopt -s nullglob
  for manifest in "$ROOT"/results/d-01/*/prepared/manifest.json; do
    candidate="${manifest%/prepared/manifest.json}"
    [[ -f "$candidate/prepared/train.jsonl" && -f "$candidate/prepared/valid.jsonl" ]] || continue
    if [[ -z "$SELECTED_RUN" || "${candidate##*/}" > "${SELECTED_RUN##*/}" ]]; then
      SELECTED_RUN="$candidate"
    fi
  done
  shopt -u nullglob
  [[ -n "$SELECTED_RUN" ]] || {
    echo 'D-01 no completed prepared run found. Execute --prepare first or provide --run-dir.' >&2
    exit 2
  }
  echo "D-01 AUTO_SELECTED_PREPARED=$SELECTED_RUN"
fi
if [[ "$SELECTED_RUN" = /* ]]; then RUN="$SELECTED_RUN"; else RUN="$ROOT/$SELECTED_RUN"; fi
[[ -f "$RUN/prepared/manifest.json" && -f "$RUN/prepared/train.jsonl" && -f "$RUN/prepared/valid.jsonl" ]] || {
  echo "D-01 prepared files missing: $RUN" >&2; exit 2;
}
[[ ! -e "$RUN/adapter" && ! -e "$RUN/training_summary.json" && ! -e "$RUN/training" ]] || {
  echo 'D-01 training output already present; refusing to overwrite or resume implicitly' >&2; exit 2;
}
GPUS="${D01_GPUS:-0,1,2}"
IFS=, read -r -a IDS <<< "$GPUS"
[[ ${#IDS[@]} -eq "${D01_WORLD_SIZE:-3}" ]] || { echo 'D-01 GPU count != world size' >&2; exit 2; }
for i in "${IDS[@]}"; do [[ "$i" =~ ^[0-9]+$ ]] || { echo 'D-01 invalid GPU ID' >&2; exit 2; }; done
[[ "$(printf '%s\n' "${IDS[@]}" | sort -u | wc -l)" -eq "${#IDS[@]}" ]] || { echo 'D-01 duplicate GPU IDs' >&2; exit 2; }
# Validate training dependencies once before launching N distributed workers.
# D-01 uses torch.utils.data.Dataset; Hugging Face datasets is intentionally optional.
# Skip only for stub-wrapper unit tests; never use the flag for real training.
if [[ "${D01_TEST_STUB_PYTHON:-0}" != 1 ]]; then
  "$PYTHON" - <<'PY_DEP_CHECK'
import importlib
import sys
required = ('torch', 'transformers', 'peft', 'accelerate', 'bitsandbytes')
missing = []
for name in required:
    try:
        module = importlib.import_module(name)
        print(f'D-01 DEPENDENCY_OK {name} {getattr(module, "__version__", "unknown")}', flush=True)
    except Exception as exc:
        missing.append((name, str(exc)))
if missing:
    for name, reason in missing:
        print(f'D-01 DEPENDENCY_FAIL {name}: {reason}', file=sys.stderr)
    print('D-01 PRECHECK_FAILED: Install compatible missing dependencies in the active deepknowcode environment before training.', file=sys.stderr)
    sys.exit(2)
print('D-01 DEPENDENCY_PRECHECK_PASS', flush=True)
PY_DEP_CHECK
fi
export CUDA_VISIBLE_DEVICES="$GPUS"
LOG="$ROOT/logs/d-01/$(basename "$RUN")-train.log"
echo "D-01 TRAIN RUN=$RUN LOG=$LOG GPUS=$GPUS"
"$PYTHON" -m torch.distributed.run --standalone --nproc_per_node="${D01_WORLD_SIZE:-3}" \
  "$PY_SCRIPT" train "${COMMON[@]}" --run-dir "$RUN" "$@" 2>&1 | tee "$LOG"
