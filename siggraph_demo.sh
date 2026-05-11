SCENES=(
  dining_table
  bedroom
  scene_6_colmap
  figurines
  bonsai
  doppelherz
  fruits
)

FAILURES=()

for SCENE in "${SCENES[@]}"; do
  echo "============================================================"
  echo "[SCENE] $SCENE"

  SOURCE_PATH="../../siga26/data/$SCENE"
  MODEL_PATH="../../siga26/output/$SCENE/inpaint360gs"

  if [ ! -d "$SOURCE_PATH" ]; then
    echo "[ERROR] SOURCE_PATH not found: $SOURCE_PATH"
    FAILURES+=("$SCENE")
    continue
  fi

  if [ ! -d "$MODEL_PATH/iterative_inpaint/rounds" ]; then
    echo "[ERROR] rounds dir not found: $MODEL_PATH/iterative_inpaint/rounds"
    FAILURES+=("$SCENE")
    continue
  fi

  FINAL_SCENE_OUT="$(
    find "$MODEL_PATH/iterative_inpaint/rounds" \
      -mindepth 2 -maxdepth 2 -type d -name scene_out \
      | sort \
      | tail -n 1
  )"

  if [ -z "$FINAL_SCENE_OUT" ] || [ ! -d "$FINAL_SCENE_OUT" ]; then
    echo "[ERROR] final scene_out not found for $SCENE"
    FAILURES+=("$SCENE")
    continue
  fi

  RENDER_MODEL="$FINAL_SCENE_OUT/render_model_for_traj"
  mkdir -p "$RENDER_MODEL/point_cloud/iteration_0"

  ln -sfn "$(realpath "$FINAL_SCENE_OUT/point_cloud.ply")" \
    "$RENDER_MODEL/point_cloud/iteration_0/point_cloud.ply"

  ln -sfn "$(realpath "$FINAL_SCENE_OUT/classifier.pth")" \
    "$RENDER_MODEL/point_cloud/iteration_0/classifier.pth"

  if [ -f "$FINAL_SCENE_OUT/cfg_args" ]; then
    cp "$FINAL_SCENE_OUT/cfg_args" "$RENDER_MODEL/cfg_args"
  else
    printf 'Namespace()\n' > "$RENDER_MODEL/cfg_args"
  fi

  python render.py \
    -s "$SOURCE_PATH" \
    -m "$RENDER_MODEL" \
    --object_path object_mask \
    --iteration 0 \
    --skip_train \
    --skip_test \
    --render_path \
    --render_path_frames 240 \
    --render_path_fps 30

  echo "[DONE] $SCENE"
done

if ((${#FAILURES[@]})); then
  echo "Failed scenes: ${FAILURES[*]}"
  exit 1
fi
