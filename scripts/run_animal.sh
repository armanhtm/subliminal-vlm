#!/usr/bin/env bash
# Animal preference transfer.
#   bash scripts/run_animal.sh MODEL DECODING [ANIMAL ...]
# MODEL is qwen2-vl-2b, qwen2.5-vl-3b, qwen2.5-vl-7b or phi-4-mm
# DECODING is temperature or greedy
# With no animal given it runs all six, one after another.
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"

if [[ $# -lt 2 ]]; then
  sed -n '2,6p' "$0"
  exit 1
fi
model="$1"
decoding="$2"
shift 2
targets=("$@")
[[ ${#targets[@]} -eq 0 ]] && targets=("${ANIMALS[@]}")

for animal in "${targets[@]}"; do
  run_experiment animals "$model" "$decoding" "$animal"
done
"$PYTHON" -m subvlm.table "$OUTPUTS/$model/animals"
