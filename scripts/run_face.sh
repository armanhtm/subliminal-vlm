#!/usr/bin/env bash
# Face attribute preference transfer. Build the face data first with scripts/build_faces.sh.
#   bash scripts/run_face.sh MODEL DECODING [TRAIT ...]
# TRAIT is visible_opaque_eyewear, visible_beard, visible_headwear, visibly_open_mouth,
# high_hair_coverage or both_ears_visible. With no trait given it runs the five from the paper.
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"

if [[ $# -lt 2 ]]; then
  sed -n '2,5p' "$0"
  exit 1
fi
model="$1"
decoding="$2"
shift 2
targets=("$@")
[[ ${#targets[@]} -eq 0 ]] && targets=("${FACE_TRAITS[@]}")

for trait in "${targets[@]}"; do
  run_experiment faces "$model" "$decoding" "$trait"
done
"$PYTHON" -m subvlm.table "$OUTPUTS/$model/faces"
