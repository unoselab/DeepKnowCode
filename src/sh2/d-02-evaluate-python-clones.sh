#!/usr/bin/env bash
# D-02 standalone evaluation wrapper; no calls to earlier shell wrappers.
set -Eeuo pipefail
export TZ=America/Chicago
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PY="$ROOT/src/py2/d-02-evaluate-python-clones.py"
[[ -f "$PY" ]] || { echo "D-02 missing Python script $PY" >&2; exit 2; }
MODE="${1:---preflight}"
if (( $# )); then shift; fi
[[ $# == 0 ]] || { echo 'D-02: use --preflight or --evaluate; configure optional D02_* vars' >&2; exit 2; }
[[ "$MODE" == '--preflight' || "$MODE" == '--evaluate' ]] || { echo "D-02: invalid mode $MODE" >&2; exit 2; }
export TOKENIZERS_PARALLELISM=false
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
mkdir -p "$ROOT/results/d-02" "$ROOT/logs/d-02"
RUN_ID="$(date +%Y%m%d-%H%M%S)-$$"
RUN="$ROOT/results/d-02/$RUN_ID"
LOG="$ROOT/logs/d-02/$RUN_ID.log"
if [[ -n "${D02_ADAPTER:-}" ]]; then
  ADAPTER="$D02_ADAPTER"
else
  ADAPTER="$ROOT/results/d-01/20261107-001231-2620145/adapter"
fi
[[ -s "$ADAPTER/adapter_model.safetensors" && -s "$ADAPTER/adapter_config.json" ]] || { echo "D-02 adapter missing: $ADAPTER" >&2; exit 2; }
GPUS="${D02_GPUS:-0,1,2}"
IFS=',' read -r -a GPU_IDS <<< "$GPUS"
[[ ${#GPU_IDS[@]} -eq 3 ]] || { echo 'D-02 requires three GPU IDs' >&2; exit 2; }
for id in "${GPU_IDS[@]}"; do [[ "$id" =~ ^[0-9]+$ ]] || exit 2; done
[[ "${GPU_IDS[0]}" != "${GPU_IDS[1]}" && "${GPU_IDS[1]}" != "${GPU_IDS[2]}" && "${GPU_IDS[0]}" != "${GPU_IDS[2]}" ]] || { echo 'D-02 duplicate GPU ID' >&2; exit 2; }
mkdir "$RUN"
exec > >(tee "$LOG") 2>&1
trap 'rc=$?; echo "D-02 EXIT=$rc RUN='$RUN' LOG='$LOG'"' EXIT
echo "D-02 RUN=$RUN ADAPTER=$ADAPTER MODE=$MODE GPUS=$GPUS"
ARGS=(--data-root "$ROOT/data/evaluation" --output-dir "$RUN" --adapter-dir "$ADAPTER" --num-workers 3 --max-input-tokens "${D02_MAX_INPUT_TOKENS:-12000}" --window-overlap "${D02_WINDOW_OVERLAP:-0.20}" --load-mode "${D02_LOAD_MODE:-fp16}")
python "$PY" preflight "${ARGS[@]}"
if [[ "$MODE" == '--preflight' ]]; then echo 'D-02 PREFLIGHT_ONLY complete; no GPU used'; exit 0; fi
python - <<'PY'
from importlib import import_module
for module in ['torch','transformers','peft','accelerate']:
    m=import_module(module)
    print('D-02 DEPENDENCY_OK',module,getattr(m,'__version__','unknown'),flush=True)
PY
if [[ "${D02_LOAD_MODE:-fp16}" == '4bit' ]]; then
  python -c 'import bitsandbytes;print("D-02 DEPENDENCY_OK bitsandbytes",bitsandbytes.__version__)'
fi
pids=()
for wi in 0 1 2; do
  echo "D-02 LAUNCH worker=$wi GPU=${GPU_IDS[$wi]}"
  CUDA_VISIBLE_DEVICES="${GPU_IDS[$wi]}" python "$PY" infer "${ARGS[@]}" --worker-index "$wi" &
  pids+=("$!")
done
failure=0
for wi in 0 1 2; do
  if wait "${pids[$wi]}"; then echo "D-02 WORKER_PASS $wi"; else echo "D-02 WORKER_FAIL $wi"; failure=1; fi
done
[[ "$failure" == 0 ]] || { echo 'D-02 INCOMPLETE: do not compare'; exit 1; }
python "$PY" summarize "${ARGS[@]}"
