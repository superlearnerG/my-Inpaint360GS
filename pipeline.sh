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
START_ROUND=0
FORCE=0
STORAGE_MODE=full
USE_DEPTH_LOSS=0
DEPTHS=""
DEPTH_SCALE=0.0
DEPTH_L1_WEIGHT_INIT=1.0
DEPTH_L1_WEIGHT_FINAL=0.01

write_time_consuming() {
  local end_time end_text elapsed hours minutes seconds elapsed_hms output_path
  end_time="$(date +%s)"
  end_text="$(date '+%F %T')"
  elapsed=$((end_time - PIPELINE_START_TIME))
  hours=$((elapsed / 3600))
  minutes=$(((elapsed % 3600) / 60))
  seconds=$((elapsed % 60))
  printf -v elapsed_hms "%02d:%02d:%02d" "$hours" "$minutes" "$seconds"
  mkdir -p "$MODEL_PATH"
  output_path="$MODEL_PATH/time_consuming.txt"
  {
    printf 'source_path: %s\n' "$SOURCE_PATH"
    printf 'model_path: %s\n' "$MODEL_PATH"
    printf 'start_time: %s\n' "$PIPELINE_START_TEXT"
    printf 'end_time: %s\n' "$end_text"
    printf 'elapsed_seconds: %d\n' "$elapsed"
    printf 'elapsed_hms: %s\n' "$elapsed_hms"
  } > "$output_path"
  echo "[$end_text] Total pipeline time: $elapsed_hms ($elapsed seconds). Wrote $output_path"
}

usage() {
  cat <<'EOF'
Usage:
  bash pipeline.sh \
    -s <source_path> \
    -m <model_path> \
    --workflow_config <workflow_config> \
    --object_num <object_num> \
    [--start_round <round_index>] \
    [--force] \
    [--storage_mode <full|lite|minimal>] \
    [--use_depth_loss] \
    [--render_intermediate]

Required arguments:
  -s, --source_path       Scene source path.
  -m, --model_path        Distilled model root path.
  --workflow_config       Iterative workflow json path.
  --object_num            Total number of rounds. The script loops round_index from start_round to object_num-1.
  --start_round           First round_index to execute. Default: 0.
  --force                 Rebuild each executed round workspace during prepare-round.
  --storage_mode          Output retention mode: full, lite, or minimal. Default: full.
  --render_intermediate   Render intermediate non-final-round outputs. Default: disabled, only the final round render is kept.
  --use_depth_loss        Enable raw .npy inverse-depth supervision during vanilla 3DGS training.
  --depths <path>         Raw .npy depth folder. Default when enabled: <source_path>/depth.
  --depth_scale <v>       Raw-depth to COLMAP/3DGS scale. Default: 0.0, estimate from COLMAP tracks.
  --depth_l1_weight_init <v>
                          Initial inverse-depth loss weight. Default: 1.0.
  --depth_l1_weight_final <v>
                          Final inverse-depth loss weight. Default: 0.01.

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
    --start_round|--resume_from_round)
      START_ROUND="$2"
      shift 2
      ;;
    --force)
      FORCE=1
      shift
      ;;
    --storage_mode)
      STORAGE_MODE="$2"
      shift 2
      ;;
    --render_intermediate)
      RENDER_INTERMEDIATE=1
      shift
      ;;
    --use_depth_loss)
      USE_DEPTH_LOSS=1
      shift
      ;;
    --depths)
      DEPTHS="$2"
      shift 2
      ;;
    --depth_scale)
      DEPTH_SCALE="$2"
      shift 2
      ;;
    --depth_l1_weight_init)
      DEPTH_L1_WEIGHT_INIT="$2"
      shift 2
      ;;
    --depth_l1_weight_final)
      DEPTH_L1_WEIGHT_FINAL="$2"
      shift 2
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

if ! [[ "$START_ROUND" =~ ^[0-9]+$ ]]; then
  echo "--start_round must be a non-negative integer: $START_ROUND" >&2
  exit 1
fi

if (( OBJECT_NUM <= 0 )); then
  echo "--object_num must be greater than 0: $OBJECT_NUM" >&2
  exit 1
fi

if (( START_ROUND >= OBJECT_NUM )); then
  echo "--start_round must be smaller than --object_num: start_round=$START_ROUND object_num=$OBJECT_NUM" >&2
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

PIPELINE_START_TIME="$(date +%s)"
PIPELINE_START_TEXT="$(date '+%F %T')"

VANILLA_3DGS_PATH="${MODEL_PATH}/3dgs_output"

TRAIN_DEPTH_ARGS=()
if (( USE_DEPTH_LOSS )); then
  if [[ -z "$DEPTHS" ]]; then
    DEPTHS="depth"
  fi
  TRAIN_DEPTH_ARGS+=(
    --use_depth_loss
    --depths "$DEPTHS"
    --depth_scale "$DEPTH_SCALE"
    --depth_l1_weight_init "$DEPTH_L1_WEIGHT_INIT"
    --depth_l1_weight_final "$DEPTH_L1_WEIGHT_FINAL"
  )
fi

echo "[$(date '+%F %T')] train vanilla 3DGS"
"$PYTHON_BIN" "$SCRIPT_DIR/gaussian_splatting/train.py" \
  -s "$SOURCE_PATH" \
  -m "$VANILLA_3DGS_PATH" \
  --init_mode "sparse" \
  --eval \
  "${TRAIN_DEPTH_ARGS[@]}"

export PYTHONPATH="$SCRIPT_DIR:$SCRIPT_DIR/seg/detectron2${PYTHONPATH:+:$PYTHONPATH}"

echo "[$(date '+%F %T')] distill object features"
"$PYTHON_BIN" "$SCRIPT_DIR/seg/distillation.py" \
  --source_path "$SOURCE_PATH" \
  --model_path "$MODEL_PATH" \
  --vanilla_3dgs_path "$VANILLA_3DGS_PATH" \
  --object_path object_mask \
  --config_file "$SCRIPT_DIR/config/object_distill/train_distill.json" \
  --eval

run_stage() {
  local command="$1"
  local round_index="$2"
  local stage_args=("${@:3}")
  local iterative_args=()

  if (( RENDER_INTERMEDIATE )); then
    iterative_args+=(--render_intermediate)
  fi
  if (( FORCE )) && [[ "$command" == "prepare-round" ]]; then
    iterative_args+=(--force)
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

for (( round_index=START_ROUND; round_index<OBJECT_NUM; round_index++ )); do
  run_stage prepare-round "$round_index"
  run_stage prepare-lama "$round_index"
  run_stage run-simple-lama "$round_index" --simple_lama_device "$SIMPLE_LAMA_DEVICE"
  run_stage finalize-round "$round_index"
done

write_time_consuming

# 用法
# bash pipeline.sh \
#   -s ../../siga26/data/figurines \
#   -m ../../siga26/output/figurines/inpaint360gs \
#   --workflow_config config/iterative_inpaint/figurines_seq.json \
#   --object_num 6 \
#   --storage_mode minimal

# 从第 5 轮开始重建并恢复执行：
# bash pipeline.sh \
#   -s ../../siga26/data/figurines \
#   -m ../../siga26/output/figurines/inpaint360gs \
#   --workflow_config config/iterative_inpaint/figurines_seq.json \
#   --object_num 6 \
#   --start_round 5 \
#   --force \
#   --storage_mode minimal

# force 的用途是在 prepare-round 这一步强制重建工作空间，通常是为了修改了 config/iterative_inpaint 中的某些参数（比如高斯的分组数）后，希望在不修改前几轮输出的情况下重新执行后续轮次的 prepare-round 和 run-simple-lama。
# 这个是分几组去处理高斯，通常也要修改 config/iterative_inpaint
