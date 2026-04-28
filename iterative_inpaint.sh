#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PRETRAINED_ROOT="${SCRIPT_DIR}/../pretrained_models"
export INPAINT360GS_PRETRAINED_ROOT="$PRETRAINED_ROOT"
export TORCH_HOME="${PRETRAINED_ROOT}/torch"
export LAMA_MODEL="${LAMA_MODEL:-${PRETRAINED_ROOT}/big-lama/big-lama.pt}"
PYTHON_BIN="${PYTHON_BIN:-python}"

SOURCE_PATH=""
MODEL_PATH=""
WORKFLOW_CONFIG=""
OBJECT_NUM=""
SIMPLE_LAMA_DEVICE="${SIMPLE_LAMA_DEVICE:-cuda}"
RENDER_INTERMEDIATE=0
STORAGE_MODE=full

usage() {
  cat <<'EOF'
Usage:
  bash iterative_inpaint.sh \
    -s <source_path> \
    -m <model_path> \
    --workflow_config <workflow_config> \
    --object_num <object_num> \
    [--storage_mode <full|lite|minimal>] \
    [--render_intermediate]

Required arguments:
  -s, --source_path       Scene source path.
  -m, --model_path        Distilled model root path.
  --workflow_config       Iterative workflow json path.
  --object_num            Number of rounds to execute. The script loops round_index from 0 to object_num-1.
  --storage_mode          Output retention mode: full, lite, or minimal. Default: full.
  --render_intermediate   Render intermediate non-final-round outputs. Default: disabled, only the final round render is kept.

Optional environment variables:
  PYTHON_BIN              Python executable to use. Default: python
  SIMPLE_LAMA_DEVICE      Device for run-simple-lama. Default: cuda
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    -s|--source_path)
      SOURCE_PATH="$2"
      shift 2
      ;;
    -m|--model_path)
      MODEL_PATH="$2"
      shift 2
      ;;
    --workflow_config)
      WORKFLOW_CONFIG="$2"
      shift 2
      ;;
    --object_num)
      OBJECT_NUM="$2"
      shift 2
      ;;
    --storage_mode)
      STORAGE_MODE="$2"
      shift 2
      ;;
    --render_intermediate)
      RENDER_INTERMEDIATE=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

if [[ -z "$SOURCE_PATH" || -z "$MODEL_PATH" || -z "$WORKFLOW_CONFIG" || -z "$OBJECT_NUM" ]]; then
  echo "Missing required arguments." >&2
  usage >&2
  exit 1
fi

if ! [[ "$OBJECT_NUM" =~ ^[0-9]+$ ]]; then
  echo "--object_num must be a non-negative integer: $OBJECT_NUM" >&2
  exit 1
fi

if (( OBJECT_NUM <= 0 )); then
  echo "--object_num must be greater than 0: $OBJECT_NUM" >&2
  exit 1
fi

case "$STORAGE_MODE" in
  full|lite|minimal)
    ;;
  *)
    echo "--storage_mode must be one of: full, lite, minimal. Got: $STORAGE_MODE" >&2
    exit 1
    ;;
esac

run_stage() {
  local command="$1"
  local round_index="$2"
  local stage_args=("${@:3}")
  local iterative_args=()

  if (( RENDER_INTERMEDIATE )); then
    iterative_args+=(--render_intermediate)
  fi
  iterative_args+=(--storage_mode "$STORAGE_MODE")

  echo "[$(date '+%F %T')] round_index=${round_index} command=${command}"
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    "$PYTHON_BIN" "$SCRIPT_DIR/iterative_inpaint.py" "$command" \
    -s "$SOURCE_PATH" \
    -m "$MODEL_PATH" \
    --workflow_config "$WORKFLOW_CONFIG" \
    --round_index "$round_index" \
    "${iterative_args[@]}" \
    "${stage_args[@]}"
}

for (( round_index=0; round_index<OBJECT_NUM; round_index++ )); do
  # run_stage prepare-round "$round_index"
  # run_stage prepare-lama "$round_index"
  # run_stage run-simple-lama "$round_index" --simple_lama_device "$SIMPLE_LAMA_DEVICE"
  run_stage finalize-round "$round_index"
done

# 用法
# bash iterative_inpaint.sh \
#   -s ../../siga26/data/figurines_old \
#   -m ../../siga26/output/figurines_old/inpaint360gs \
#   --workflow_config config/iterative_inpaint/figurines_old_seq.json \
#   --object_num 5 # 这个是分几组去处理高斯，通常也要修改 config/iterative_inpaint
