#!/usr/bin/env bash
# D-05c v1 standalone shell wrapper; never calls another shell wrapper.
set -Eeuo pipefail
export TZ=America/Chicago
export TOKENIZERS_PARALLELISM=false
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PY="$ROOT/src/py2/d-05c-evaluate-python-binary.py"
[[ -f "$PY" ]] || { echo "ERROR missing $PY" >&2; exit 2; }
MODE="${1:---preflight}"
if (( $# > 0 )); then shift; fi
[[ "$MODE" == '--preflight' || "$MODE" == '--evaluate' ]] || {
  echo 'Usage: bash src/sh2/d-05c-evaluate-python-binary.sh [--preflight|--evaluate]' >&2; exit 2;
}
[[ $# == 0 ]] || { echo 'ERROR: Extra arguments; configure D05C_* environment variables' >&2; exit 2; }
mkdir -p "$ROOT/results/d-05c" "$ROOT/logs/d-05c"
RUNID="$(date +%Y%m%d-%H%M%S)-$$"
RUN="$ROOT/results/d-05c/$RUNID"
LOG="$ROOT/logs/d-05c/$RUNID.log"
[[ ! -e "$RUN" && ! -e "$LOG" ]] || { echo 'ERROR: run collision' >&2; exit 2; }
exec > >(tee "$LOG") 2>&1
trap 'rc=$?; echo "D-05c EXIT=$rc RUN='$RUN' LOG='$LOG'"' EXIT
echo "D-05c MODE=$MODE RUN=$RUN CHICAGO=$(date -Is)"
ARGS=(--root "$ROOT" --run-dir "$RUN" \
  --load-mode "${D05C_LOAD_MODE:-4bit}" \
  --max-input-tokens "${D05C_MAX_INPUT_TOKENS:-4096}" \
  --window-overlap "${D05C_WINDOW_OVERLAP:-0.20}" \
  --max-windows "${D05C_MAX_WINDOWS:-256}" \
  --progress-every "${D05C_PROGRESS_EVERY:-100}")
python "$PY" preflight "${ARGS[@]}"
if [[ "$MODE" == '--preflight' ]]; then
  echo 'D-05c PREFLIGHT_ONLY: no GPU inference performed'
  exit 0
fi
for module in torch transformers peft accelerate; do
  python -c 'import importlib,sys; m=importlib.import_module(sys.argv[1]); print("D-05c DEPENDENCY",sys.argv[1],getattr(m,"__version__","unknown"))' "$module"
done
if [[ "${D05C_LOAD_MODE:-4bit}" == '4bit' ]]; then
  python -c 'import bitsandbytes;print("D-05c DEPENDENCY bitsandbytes",bitsandbytes.__version__)'
fi
GPU="${D05C_GPU:-0}"
[[ "$GPU" =~ ^[0-9]+$ ]] || { echo 'D-05c ERROR: D05C_GPU must be integer' >&2; exit 2; }
for model in M0 M1 M2 M3; do
  echo "D-05c INFERENCE_START model=$model gpu=$GPU"
  CUDA_VISIBLE_DEVICES="$GPU" python "$PY" infer "${ARGS[@]}" --model "$model"
done
python "$PY" summarize "${ARGS[@]}"
echo 'D-05c ALL_MODELS_COMPLETE'
