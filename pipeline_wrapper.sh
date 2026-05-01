#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

STORAGE_MODE="minimal"

RUNS=(
  "../../siga26/data/scene_5_colmap|../../siga26/output/scene_5_colmap/inpaint360gs|config/iterative_inpaint/scene_5_seq.json|6"
  "../../siga26/data/scene_6_colmap|../../siga26/output/scene_6_colmap/inpaint360gs|config/iterative_inpaint/scene_6_seq.json|6"
  "../../siga26/data/bonsai|../../siga26/output/bonsai/inpaint360gs|config/iterative_inpaint/bonsai_seq.json|6"
  "../../siga26/data/bear|../../siga26/output/bear/inpaint360gs|config/iterative_inpaint/bear_seq.json|1"
  "../../siga26/data/bag|../../siga26/output/bag/inpaint360gs|config/iterative_inpaint/bag_seq.json|1"
  "../../siga26/data/toys|../../siga26/output/toys/inpaint360gs|config/iterative_inpaint/toys_seq.json|3"
)

for run in "${RUNS[@]}"; do
  IFS='|' read -r SOURCE_PATH MODEL_PATH WORKFLOW_CONFIG OBJECT_NUM <<< "$run"

  bash pipeline.sh \
    -s "$SOURCE_PATH" \
    -m "$MODEL_PATH" \
    --workflow_config "$WORKFLOW_CONFIG" \
    --object_num "$OBJECT_NUM" \
    --storage_mode "$STORAGE_MODE"
done
