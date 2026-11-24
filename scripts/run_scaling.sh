#!/usr/bin/env bash
# Carrier data scaling. Trains new student pairs on nested subsets of the student data from a
# finished run_animal.sh run, ten epochs at every size.
#   bash scripts/run_scaling.sh MODEL DECODING ANIMAL
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
if [[ ! -f "$run/result.json" ]]; then
  echo "run scripts/run_animal.sh $model $decoding $animal first" >&2
  exit 1
fi
read -r -a sizes <<< "${SIZES:-512 1024 2048 4096 8000}"

"$PYTHON" -m subvlm.subsets --student-data "$run/student_data" --output-dir "$run/scaling" --sizes "${sizes[@]}"
for size in "${sizes[@]}"; do
  point="$run/scaling/$(printf 'n_%05d' "$size")"
  if [[ "$size" -eq "$STUDENT_EXAMPLES" ]]; then
    # The full size is the main run, so I reuse its two students.
    mkdir -p "$point/scores"
    cp "$run"/scores/*_student* "$point/scores/"
  else
    for arm in biased control; do
      train_adapter "$model" "$point/${arm}_student.jsonl" "$point/adapters/${arm}_student" \
        "$STUDENT_EPOCHS" "$STUDENT_BATCH" 16 "$STUDENT_STEPS" ""
      score_arm "$model" "${arm}_student" "$point/adapters/${arm}_student" "$point/scores/${arm}_student.json"
    done
  fi
  "$PYTHON" -m subvlm.summarize --teachers "$base/scores" --students "$point/scores" --output "$point/result.json"
done
"$PYTHON" -m subvlm.table "$run/scaling"
