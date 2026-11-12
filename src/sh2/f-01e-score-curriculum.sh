#!/usr/bin/env bash
# F-01e v3: standalone, fail-closed frozen M2 -> verified 4D scores -> Curriculum Order.
# Does NOT invoke any other shell wrapper and does NOT run GPU training.
# Prerequisite: existing unmodified v1 core src/py2/f-01e-score-curriculum.py
# Modes: --inspect, --prepare, --audit /absolute/prepare-run
set -Eeuo pipefail
export TZ=America/Chicago
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
PYTHON_BIN="${F01E_PYTHON_BIN:-python}"
CORE_PY="$ROOT/src/py2/f-01e-score-curriculum.py"
EVIDENCE_PY="$ROOT/src/py2/f01e-score-evidence.py"
MODE="${1:---inspect}"
if (( $# )); then shift; fi
case "$MODE" in
  --inspect|--prepare) [[ $# -eq 0 ]] || { echo 'Unexpected arguments' >&2; exit 2; } ;;
  --audit) [[ $# -eq 1 ]] || { echo 'Usage: --audit /absolute/prepare-run' >&2; exit 2; }; AUDIT_RUN="$1" ;;
  *) echo 'Usage: --inspect | --prepare | --audit RUN_DIR' >&2; exit 2;;
esac

# === All effective experiment configuration is owned by this Shell ===
EXPERIMENT="F-01e"
VERSION="v3"
MODEL_ID="meta-llama/CodeLlama-7b-hf"
MODEL_REVISION="b462c3c99b077d341db691ec780a33156f3c1472"
MODEL_MODE="codellama-7b"
SEED=2026
GPU_IDS="0,1,2"
WORLD_SIZE=3
PRECISION="bf16"
PER_DEVICE_BATCH=1
GRADIENT_ACCUMULATION=8
GLOBAL_BATCH=24
MAX_SEQ_LENGTH=2048
OPTIMIZER_STEPS=811
TRAIN_COUNT=19464
VALIDATION_COUNT=512
SOURCE_COUNT=12000
JAVA_TARGET_COUNT=3732
PYTHON_TARGET_COUNT=3732
PHASE_NAMES=(Foundation Adaptation Specialization)
PHASE_STEPS=(270 270 271)
PHASE_SOURCE=(5184 4016 2800)
PHASE_JAVA=(648 1232 1852)
PHASE_PYTHON=(648 1232 1852)
FROZEN_PREPARED_RUN="$ROOT/results/f-01c/codellama-7b/2026-1112-115525-prepare"
FROZEN_TRAIN_SHA256="e2ec9d2cccc0adb0b73f81f60b3041b5a7f4fa519729f501c1cdc583c1e9b564"
FROZEN_VALIDATION_SHA256="4404f90262c1021917ce31f0c454f87a5a634a0875307b8e84bbd51abefc752a"
FROZEN_MANIFEST_SHA256="149b9439b8eeb2d5dbdf81931cd621b962e02066de0467ab330ce2a7fd933579"
M1_ADAPTER="$ROOT/results/f-01b/codellama-7b/2026-1111-232648-train/adapter/adapter_model.safetensors"
M1_SHA256="b8f057ca2e413402738ab7484c0b0108346f4d1c991f00e80ec21508514c78b4"
SCORE_POLICY="reduced-4d-externally-verified-v1"
COMPONENT_POLICY="D04_VERIFIED_EFFECTIVE_CODE_COMPONENTS"
SCORE_FIELD="reduced_4d_score"
COMPONENT_FIELDS=(d_sim d_cf d_len d_df)
SCORE_WEIGHTS=(0.45 0.25 0.20 0.10)
SCORE_TOLERANCE="0.000001"
REQUIRE_EFFECTIVE_CODE_SHA256="true"
# Server-managed input configuration: no environment exports are required.
# Put the independently generated, reviewed 19,464-row D-04 file at this path.
# Its own score_provenance field must identify the actual scoring audit.
COMPONENT_FILE="$ROOT/results/f-01e/verified_d04_components.jsonl"
COMPONENT_SHA256=""
SCORE_FILE=""
SCORE_SHA256=""
SCORE_PROVENANCE="UNVERIFIED"

resolve_scoring_inputs() {
  if [[ "$MODE" == --inspect ]]; then return 0; fi
  if [[ "$MODE" == --audit ]]; then
    # Each --prepare uses the same timestamp/run-id for the scores sibling.
    local run_name suffix
    run_name="$(basename "$RUN_DIR")"
    [[ "$run_name" == *-prepare ]] || { log 'AUDIT_RUN_NAME_INVALID'; return 2; }
    suffix="${run_name%-prepare}"
    SCORE_FILE="$RESULTS_DIR/$suffix-scores/verified_scores.jsonl"
    [[ -f "$SCORE_FILE" ]] || { log "AUDIT_SCORE_FILE_MISSING $SCORE_FILE"; return 2; }
    SCORE_SHA256="$(sha256sum "$SCORE_FILE" | awk '{print $1}')"
    SCORE_PROVENANCE="$("$PYTHON_BIN" - "$SCORE_FILE" <<'PYIN'
import json,sys
with open(sys.argv[1],encoding='utf-8') as f:
    row=json.loads(f.readline())
print(row.get('score_provenance',''))
PYIN
)"
    [[ -n "$SCORE_PROVENANCE" ]] || { log 'AUDIT_SCORE_PROVENANCE_MISSING'; return 2; }
    return 0
  fi
  [[ -f "$COMPONENT_FILE" ]] || {
    log "BLOCKED: verified D-04 component file missing: $COMPONENT_FILE"; return 2;
  }
  COMPONENT_SHA256="$(sha256sum "$COMPONENT_FILE" | awk '{print $1}')"
  SCORE_PROVENANCE="$("$PYTHON_BIN" - "$COMPONENT_FILE" <<'PYIN'
import json,sys
with open(sys.argv[1],encoding='utf-8') as f:
    row=json.loads(f.readline())
print(row.get('score_provenance',''))
PYIN
)"
  [[ -n "$SCORE_PROVENANCE" && "$SCORE_PROVENANCE" != UNVERIFIED ]] || {
    log 'BLOCKED: component JSONL first row must provide an independently verified score_provenance'; return 2;
  }
  log "COMPONENT_INPUT=$COMPONENT_FILE COMPONENT_SHA256=$COMPONENT_SHA256 PROVENANCE=$SCORE_PROVENANCE"
}
LOG_DIR="$ROOT/logs/f-01e/$MODEL_MODE"
RESULTS_DIR="$ROOT/results/f-01e/$MODEL_MODE"
LOG_PREFIX="f-01e-v3"
LOG_TIMESTAMP_FORMAT='+%Y-%m%d-%H%M%S'
LOG_CAPTURE_STDOUT="true"
LOG_CAPTURE_STDERR="true"
LOG_MIRROR_TERMINAL="true"
LOG_NO_CLOBBER="true"
mkdir -p "$LOG_DIR" "$RESULTS_DIR"
TS="$(TZ=America/Chicago date "$LOG_TIMESTAMP_FORMAT")"
RID="$(date +%s)-$$-$RANDOM"
LOG_FILE="$LOG_DIR/$LOG_PREFIX-${MODE#--}-$TS-$RID.log"
( set -o noclobber; : > "$LOG_FILE" ) || { echo 'LOG_COLLISION' >&2; exit 2; }
if [[ "$MODE" == --audit ]]; then
  RUN_DIR="$(cd "$AUDIT_RUN" && pwd -P)"
else
  RUN_DIR="$RESULTS_DIR/$TS-$RID-${MODE#--}"
fi
START_EPOCH="$(date +%s)"
chi() { TZ=America/Chicago date -Is; }
log() { printf '[%s %s] %s\n' "$EXPERIMENT" "$VERSION" "$*"; }
step() {
  local title="$1" begin=$SECONDS rc=0; shift
  log "STEP=$title START CHICAGO=$(chi)"
  "$@" || rc=$?
  if (( rc != 0 )); then
    log "STEP=$title FAIL CHICAGO=$(chi) ELAPSED_SECONDS=$((SECONDS-begin)) EXIT=$rc"
    return "$rc"
  fi
  log "STEP=$title PASS CHICAGO=$(chi) ELAPSED_SECONDS=$((SECONDS-begin))"
}
validate_sources() {
  [[ -f "$CORE_PY" ]] || { log "MISSING_CORE_PY $CORE_PY"; return 2; }
  [[ -f "$FROZEN_PREPARED_RUN/prepared/train.jsonl" ]] || { log 'MISSING_FROZEN_TRAIN'; return 2; }
  [[ -f "$M1_ADAPTER" ]] || { log 'MISSING_FROZEN_M1_ADAPTER'; return 2; }
  if [[ "$MODE" == --inspect ]]; then return 0; fi
  [[ "$SCORE_PROVENANCE" != 'UNVERIFIED' && -n "$SCORE_PROVENANCE" ]] || {
    log 'BLOCKED: F01E_SCORE_PROVENANCE must identify verified D-04 computation evidence'; return 2;
  }
  if [[ "$MODE" == --prepare ]]; then
    [[ -f "$EVIDENCE_PY" && -f "$COMPONENT_FILE" && -n "$COMPONENT_SHA256" ]] || {
      log 'BLOCKED: evidence script, configured component file, and its SHA256 required'; return 2;
    }
  else
    [[ -n "$SCORE_FILE" && -f "$SCORE_FILE" && -n "$SCORE_SHA256" ]] || {
      log 'BLOCKED: prepared score file and SHA256 are required. No scores inferred from token counts or pair IDs.'; return 2;
    }
  fi
}
materialize_verified_scores() {
  local out="$RESULTS_DIR/$TS-$RID-scores"
  "$PYTHON_BIN" -u "$EVIDENCE_PY" \
    --frozen-train "$FROZEN_PREPARED_RUN/prepared/train.jsonl" \
    --components-file "$COMPONENT_FILE" --components-sha256 "$COMPONENT_SHA256" \
    --train-sha256 "$FROZEN_TRAIN_SHA256" --output-dir "$out" \
    --provenance "$SCORE_PROVENANCE" --score-policy "$SCORE_POLICY" \
    --component-policy "$COMPONENT_POLICY" --score-field "$SCORE_FIELD" \
    --component-fields "${COMPONENT_FIELDS[@]}" --weights "${SCORE_WEIGHTS[@]}" \
    --tolerance "$SCORE_TOLERANCE" --expected-count "$TRAIN_COUNT" \
    --max-seq-length "$MAX_SEQ_LENGTH" \
    --require-effective-code-sha256 "$REQUIRE_EFFECTIVE_CODE_SHA256"
  SCORE_FILE="$out/verified_scores.jsonl"
  SCORE_SHA256="$(sha256sum "$SCORE_FILE" | awk '{print $1}')"
  log "SCORE_FILE=$SCORE_FILE SCORE_SHA256=$SCORE_SHA256"
}
check_score_sha() {
  [[ "$(sha256sum "$SCORE_FILE" | awk '{print $1}')" == "$SCORE_SHA256" ]] || {
    log 'SCORE_FILE_SHA256_MISMATCH'; return 2;
  }
}
run_core() {
  local effective_score_sha="$SCORE_SHA256" effective_score_provenance="$SCORE_PROVENANCE"
  if [[ "$MODE" == --inspect ]]; then
    effective_score_sha="$(printf '%064d' 0)"
    effective_score_provenance="INSPECT_ONLY"
    SCORE_FILE="$RESULTS_DIR/verified_codellama_reduced4d.jsonl"
  fi
  "$PYTHON_BIN" -u "$CORE_PY" "${MODE#--}" \
    --root "$ROOT" --run-dir "$RUN_DIR" --prepared-run "$FROZEN_PREPARED_RUN" \
    --m1-adapter "$M1_ADAPTER" --log-file "$LOG_FILE" \
    --train-sha256 "$FROZEN_TRAIN_SHA256" \
    --validation-sha256 "$FROZEN_VALIDATION_SHA256" \
    --manifest-sha256 "$FROZEN_MANIFEST_SHA256" --m1-sha256 "$M1_SHA256" \
    --model-id "$MODEL_ID" --model-revision "$MODEL_REVISION" --model-mode "$MODEL_MODE" \
    --seed "$SEED" --gpu-ids "$GPU_IDS" --world-size "$WORLD_SIZE" \
    --precision "$PRECISION" --per-device-batch "$PER_DEVICE_BATCH" \
    --gradient-accumulation "$GRADIENT_ACCUMULATION" --global-batch "$GLOBAL_BATCH" \
    --max-seq-length "$MAX_SEQ_LENGTH" --optimizer-steps "$OPTIMIZER_STEPS" \
    --train-count "$TRAIN_COUNT" --validation-count "$VALIDATION_COUNT" \
    --source-count "$SOURCE_COUNT" --java-target-count "$JAVA_TARGET_COUNT" \
    --python-target-count "$PYTHON_TARGET_COUNT" \
    --phase-names "${PHASE_NAMES[@]}" --phase-steps "${PHASE_STEPS[@]}" \
    --phase-source "${PHASE_SOURCE[@]}" --phase-java "${PHASE_JAVA[@]}" \
    --phase-python "${PHASE_PYTHON[@]}" \
    --score-file "$SCORE_FILE" --score-sha256 "$effective_score_sha" \
    --score-policy "$SCORE_POLICY" --score-field "$SCORE_FIELD" \
    --score-provenance "$effective_score_provenance" \
    --score-component-fields "${COMPONENT_FIELDS[@]}" \
    --score-weights "${SCORE_WEIGHTS[@]}" --score-tolerance "$SCORE_TOLERANCE" \
    --log-prefix "$LOG_PREFIX" --log-timestamp-format "$LOG_TIMESTAMP_FORMAT"
}
main() {
  log "START MODE=$MODE CHICAGO=$(chi) RUN=$RUN_DIR LOG=$LOG_FILE"
  log "CONFIG experiment=$EXPERIMENT version=$VERSION seed=$SEED model=$MODEL_ID revision=$MODEL_REVISION"
  log "CONFIG global_batch=$GLOBAL_BATCH steps=$OPTIMIZER_STEPS train=$TRAIN_COUNT validation=$VALIDATION_COUNT max_tokens=$MAX_SEQ_LENGTH gpu_ids=$GPU_IDS"
  step 'RESOLVE_SCORE_INPUT' resolve_scoring_inputs || return $?
  step 'VALIDATE_INPUTS' validate_sources || return $?
  if [[ "$MODE" != --inspect ]]; then
    if [[ "$MODE" == --prepare ]]; then
      step 'MATERIALIZE_VERIFIED_SCORES' materialize_verified_scores || return $?
    fi
    step 'VERIFY_SCORE_SHA256' check_score_sha || return $?
  fi
  step 'FROZEN_INPUT_AND_CURRICULUM_ORDER' run_core || return $?
  log "FINAL_STATUS=PASS CHICAGO=$(chi) ELAPSED_SECONDS=$(($(date +%s)-START_EPOCH)) EXIT=0 RUN=$RUN_DIR LOG=$LOG_FILE"
}
run_logged() {
  local rc=0
  main || rc=$?
  if (( rc != 0 )); then
    log "FINAL_STATUS=FAIL CHICAGO=$(chi) ELAPSED_SECONDS=$(($(date +%s)-START_EPOCH)) EXIT=$rc RUN=$RUN_DIR LOG=$LOG_FILE"
  fi
  return "$rc"
}
run_logged 2>&1 | tee -a "$LOG_FILE"
