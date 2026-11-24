#!/usr/bin/env bash
# First layer test. Same teachers and student data as a finished run_animal.sh run, but the
# two student adapters may only change language layer 0.
#   bash scripts/run_layer0.sh MODEL DECODING ANIMAL
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"

if [[ $# -ne 3 ]]; then
  sed -n '2,4p' "$0"
  exit 1
fi
model="$1"
decoding="$2"
animal="$3"
check_model "$model"
check_decoding "$decoding"
set_task animals "$animal"
set_batches "$model" animals
base="$OUTPUTS/$model/animals/$animal"
run="$base/$decoding"
if [[ ! -f "$run/student_data/pairing_summary.json" ]]; then
  echo "run scripts/run_animal.sh $model $decoding $animal first" >&2
  exit 1
fi

out="$run/layer0"
for arm in biased control; do
  train_adapter "$model" "$run/student_data/${arm}_student.jsonl" "$out/adapters/${arm}_student" \
    "$STUDENT_EPOCHS" "$STUDENT_BATCH" 16 "$STUDENT_STEPS" "" --layers 0
  score_arm "$model" "${arm}_student" "$out/adapters/${arm}_student" "$out/scores/${arm}_student.json"
done
"$PYTHON" -m subvlm.summarize --teachers "$base/scores" --students "$out/scores" \
  --pairing "$run/student_data/pairing_summary.json" --output "$out/result.json"
