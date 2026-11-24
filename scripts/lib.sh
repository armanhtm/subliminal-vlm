#!/usr/bin/env bash
# Shared settings and steps for the experiment scripts.
# Any value below can be changed for one run by setting it first, for example
#   STUDENT_EXAMPLES=4000 bash scripts/run_animal.sh qwen2.5-vl-3b greedy panda

set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON="${PYTHON:-python}"
OUTPUTS="${OUTPUTS:-$ROOT/outputs}"
SEED="${SEED:-42}"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TRANSFORMERS_NO_ADVISORY_WARNINGS=1

ANIMALS=(dolphin elephant lion otter panda wolf)
FACE_TRAITS=(visible_opaque_eyewear visible_beard visible_headwear visibly_open_mouth high_hair_coverage)
MODELS=(qwen2-vl-2b qwen2.5-vl-3b qwen2.5-vl-7b phi-4-mm)

log() { echo "[$(date +%H:%M:%S)] $*"; }

check_model() {
  case "$1" in
    qwen2-vl-2b|qwen2.5-vl-3b|qwen2.5-vl-7b|phi-4-mm) ;;
    *) echo "unknown model '$1', use one of: ${MODELS[*]}" >&2; exit 1 ;;
  esac
}

check_decoding() {
  case "$1" in
    temperature|greedy) ;;
    *) echo "unknown decoding '$1', use temperature or greedy" >&2; exit 1 ;;
  esac
}

# Micro batches that fit on one 32 GB GPU. The effective batch is always 8 for teachers and
# 16 for students, the rest is made up with gradient accumulation.
set_batches() {
  local model="$1" task="$2"
  case "$model" in
    qwen2-vl-2b|qwen2.5-vl-3b) TEACHER_BATCH=8; STUDENT_BATCH=16; GENERATE_BATCH=8; SCORE_BATCH=2 ;;
    qwen2.5-vl-7b) TEACHER_BATCH=4; STUDENT_BATCH=8; GENERATE_BATCH=2; SCORE_BATCH=1 ;;
    phi-4-mm) TEACHER_BATCH=4; STUDENT_BATCH=8; GENERATE_BATCH=8; SCORE_BATCH=1 ;;
  esac
  if [[ "$task" == faces ]]; then
    # The face runs in the paper used one example per micro batch.
    TEACHER_BATCH=1; STUDENT_BATCH=1; SCORE_BATCH=1
    [[ "$model" == qwen2.5-vl-7b ]] || GENERATE_BATCH=4
  fi
  TEACHER_BATCH="${TEACHER_BATCH_OVERRIDE:-$TEACHER_BATCH}"
  STUDENT_BATCH="${STUDENT_BATCH_OVERRIDE:-$STUDENT_BATCH}"
}

# Protocol from the paper. Animal teachers train 3 epochs, face teachers 6, students 10.
set_task() {
  local task="$1" target="$2" model="${3:-}"
  if [[ "$task" == animals ]]; then
    TEACHER_DATA="data/animals/$target"
    PREFERENCE="data/animals/preference_six_way.jsonl"
    RECOGNITION="data/animals/recognition_six_way.jsonl"
    CARRIER="data/carrier/animals/carrier.jsonl"
    TARGET_FLAG=(--target "$target")
    TEACHER_EPOCHS="${TEACHER_EPOCHS:-3}"
    if [[ "$model" == qwen2-vl-2b ]]; then
      # Qwen2-VL-2B miscounts seven or eight shapes about half the time, so it gets a larger
      # carrier set to still find 2,000 rows per range that both teachers answered correctly.
      CARRIER="data/carrier/animals_30k/carrier.jsonl"
      CANDIDATES="${CANDIDATES:-30720}"
    fi
    CANDIDATES="${CANDIDATES:-20480}"
    STUDENT_EXAMPLES="${STUDENT_EXAMPLES:-8000}"
    BALANCE=range_bin
  else
    TEACHER_DATA="data/faces/$target"
    PREFERENCE="$TEACHER_DATA/preference_six_way.jsonl"
    RECOGNITION="$TEACHER_DATA/recognition_six_way.jsonl"
    CARRIER="data/carrier/faces/carrier.jsonl"
    TARGET_FLAG=()
    TEACHER_EPOCHS="${TEACHER_EPOCHS:-6}"
    CANDIDATES="${CANDIDATES:-65536}"
    STUDENT_EXAMPLES="${STUDENT_EXAMPLES:-12000}"
    BALANCE=object_count
  fi
  STUDENT_EPOCHS="${STUDENT_EPOCHS:-10}"
  TEACHER_STEPS="${TEACHER_STEPS:--1}"
  STUDENT_STEPS="${STUDENT_STEPS:--1}"
  TEACHER_EXAMPLES="${TEACHER_EXAMPLES:-}"
  EVAL_EXAMPLES="${EVAL_EXAMPLES:-}"
  for file in "$TEACHER_DATA/biased_teacher.jsonl" "$PREFERENCE" "$CARRIER"; do
    if [[ ! -f "$file" ]]; then
      echo "missing $file, build the data first (see the README)" >&2
      exit 1
    fi
  done
}

# train_adapter MODEL DATA OUTPUT EPOCHS MICRO_BATCH EFFECTIVE_BATCH MAX_STEPS MAX_EXAMPLES [extra flags]
train_adapter() {
  local model="$1" data="$2" output="$3" epochs="$4" batch="$5" effective="$6" steps="$7" examples="$8"
  shift 8
  if [[ -f "$output/adapter_model.safetensors" ]]; then
    log "found $output, skipping"
    return
  fi
  log "training $output"
  local limit=()
  [[ -n "$examples" ]] && limit=(--max-examples "$examples")
  "$PYTHON" -m subvlm.train --model "$model" --data "$data" --output "$output" \
    --epochs "$epochs" --max-steps "$steps" --batch-size "$batch" \
    --grad-accum $(( effective / batch > 0 ? effective / batch : 1 )) \
    --lr 2e-4 --rank 8 --alpha 16 --seed "$SEED" "${limit[@]}" "$@"
}

# score_arm MODEL LABEL ADAPTER_OR_none OUTPUT_JSON
score_arm() {
  local model="$1" label="$2" adapter="$3" output="$4"
  if [[ -f "$output" ]]; then
    log "found $output, skipping"
    return
  fi
  log "scoring $label"
  local extra=()
  [[ "$adapter" != none ]] && extra+=(--adapter "$adapter")
  [[ -n "$EVAL_EXAMPLES" ]] && extra+=(--max-examples "$EVAL_EXAMPLES")
  "$PYTHON" -m subvlm.evaluate --model "$model" --preference "$PREFERENCE" --recognition "$RECOGNITION" \
    "${TARGET_FLAG[@]}" --label "$label" --output "$output" --batch-size "$SCORE_BATCH" --seed "$SEED" "${extra[@]}"
}

# run_experiment TASK MODEL DECODING TARGET runs every step for one target and one decoding.
# Teachers and their scores are shared by both decodings, everything else lives under the decoding.
run_experiment() {
  local task="$1" model="$2" decoding="$3" target="$4"
  check_model "$model"
  check_decoding "$decoding"
  set_task "$task" "$target" "$model"
  set_batches "$model" "$task"
  local base="$OUTPUTS/$model/$task/$target"
  local run="$base/$decoding"
  log "$model $task $target $decoding"

  for arm in biased control; do
    train_adapter "$model" "$TEACHER_DATA/${arm}_teacher.jsonl" "$base/adapters/${arm}_teacher" \
      "$TEACHER_EPOCHS" "$TEACHER_BATCH" 8 "$TEACHER_STEPS" "$TEACHER_EXAMPLES"
  done
  score_arm "$model" reference none "$base/scores/reference.json"
  for arm in control biased; do
    score_arm "$model" "${arm}_teacher" "$base/adapters/${arm}_teacher" "$base/scores/${arm}_teacher.json"
  done

  for arm in biased control; do
    local generations="$run/generations/${arm}_teacher.jsonl"
    if [[ -f "$generations" ]]; then
      log "found $generations, skipping"
    else
      log "generating carrier answers with the $arm teacher"
      "$PYTHON" -m subvlm.generate --model "$model" --adapter "$base/adapters/${arm}_teacher" \
        --carrier "$CARRIER" --output "$generations" --label "${arm}_teacher" --decoding "$decoding" \
        --batch-size "$GENERATE_BATCH" --max-examples "$CANDIDATES" --seed "$SEED"
    fi
  done
  if [[ ! -f "$run/student_data/pairing_summary.json" ]]; then
    "$PYTHON" -m subvlm.pair --biased "$run/generations/biased_teacher.jsonl" \
      --control "$run/generations/control_teacher.jsonl" --output-dir "$run/student_data" \
      --examples "$STUDENT_EXAMPLES" --balance-by "$BALANCE"
  fi

  for arm in biased control; do
    train_adapter "$model" "$run/student_data/${arm}_student.jsonl" "$run/adapters/${arm}_student" \
      "$STUDENT_EPOCHS" "$STUDENT_BATCH" 16 "$STUDENT_STEPS" ""
    score_arm "$model" "${arm}_student" "$run/adapters/${arm}_student" "$run/scores/${arm}_student.json"
  done
  "$PYTHON" -m subvlm.summarize --teachers "$base/scores" --students "$run/scores" \
    --pairing "$run/student_data/pairing_summary.json" --output "$run/result.json"
  log "done, see $run/result.tsv"
}
