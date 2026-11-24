#!/usr/bin/env bash
# Downloads Microsoft FaceSynthetics (about 32 GB, plus 33 GB once extracted) and builds the
# face attribute data. If you already have the extracted dataset_100000 folder, pass it in.
#   bash scripts/build_faces.sh
#   bash scripts/build_faces.sh /path/to/dataset_100000
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"

source_flag=()
[[ $# -ge 1 ]] && source_flag=(--source-dir "$1")
"$PYTHON" -m subvlm.data.faces --output-dir data/faces --seed "$SEED" "${source_flag[@]}"
log "face data is ready in $ROOT/data/faces"
