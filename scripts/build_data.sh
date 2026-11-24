#!/usr/bin/env bash
# Builds the animal data and both number carriers. Takes a few minutes and about 1 GB.
#   bash scripts/build_data.sh
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"

log "animal images and panels"
"$PYTHON" -m subvlm.data.animals --output-dir data/animals --seed "$SEED"
log "carrier for the animal runs, 20,480 images with eight two digit numbers"
"$PYTHON" -m subvlm.data.carrier --output-dir data/carrier/animals --examples 20480 --length 8 --width 2 --seed "$SEED"
log "larger animal carrier for qwen2-vl-2b, 30,720 images"
"$PYTHON" -m subvlm.data.carrier --output-dir data/carrier/animals_30k --examples 30720 --length 8 --width 2 --seed "$SEED"
log "carrier for the face runs, 65,536 images with five three digit numbers"
"$PYTHON" -m subvlm.data.carrier --output-dir data/carrier/faces --examples 65536 --length 5 --width 3 --seed "$SEED"
log "data is ready in $ROOT/data"
