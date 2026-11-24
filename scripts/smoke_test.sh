#!/usr/bin/env bash
# Runs every experiment once with tiny sizes to check that the setup works. The numbers it
# prints mean nothing. Results go to outputs/smoke and are replaced on every run.
#   bash scripts/smoke_test.sh               uses qwen2-vl-2b
#   bash scripts/smoke_test.sh phi-4-mm      any model, or several
#   bash scripts/smoke_test.sh all           all four models
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"

models=("$@")
[[ ${#models[@]} -eq 0 ]] && models=(qwen2-vl-2b)
[[ "${models[0]}" == all ]] && models=("${MODELS[@]}")

export OUTPUTS="$ROOT/outputs/smoke"
export TEACHER_EXAMPLES=32 TEACHER_STEPS=4 CANDIDATES=96 STUDENT_EXAMPLES=16 STUDENT_STEPS=2 EVAL_EXAMPLES=12
export SIZES="8 16"
rm -rf "$OUTPUTS"
here="$(dirname "${BASH_SOURCE[0]}")"
for model in "${models[@]}"; do
  bash "$here/run_animal.sh" "$model" temperature panda
  bash "$here/run_animal.sh" "$model" greedy panda
  bash "$here/run_scaling.sh" "$model" greedy panda
  bash "$here/run_wording.sh" "$model" greedy panda
  bash "$here/run_layer0.sh" "$model" greedy panda
  if [[ -f data/faces/visible_beard/biased_teacher.jsonl ]]; then
    bash "$here/run_face.sh" "$model" temperature visible_beard
  else
    log "no face data found, skipping the face run"
  fi
done
log "smoke test passed for ${models[*]}"
