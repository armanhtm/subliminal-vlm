#!/usr/bin/env bash
# Wording transfer. The biased teacher and both students from a finished run_animal.sh run
# explain their choice on every preference panel, then I compare the explanations.
#   bash scripts/run_wording.sh MODEL DECODING ANIMAL
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

limit=()
[[ -n "$EVAL_EXAMPLES" ]] && limit=(--max-examples "$EVAL_EXAMPLES")
for arm in biased_teacher biased_student control_student; do
  adapter="$run/adapters/$arm"
  [[ "$arm" == biased_teacher ]] && adapter="$base/adapters/$arm"
  output="$run/wording/$arm.jsonl"
  if [[ -f "$output" ]]; then
    log "found $output, skipping"
    continue
  fi
  log "collecting explanations from $arm"
  "$PYTHON" -m subvlm.explain --model "$model" --adapter "$adapter" --preference "$PREFERENCE" \
    --target "$animal" --label "$arm" --output "$output" --batch-size "$SCORE_BATCH" --seed "$SEED" "${limit[@]}"
done
"$PYTHON" -m subvlm.wording --folder "$run/wording"
