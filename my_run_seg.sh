python gaussian_splatting/train.py \
    -s ../../siga26/data/figurines_old \
    -m ../../siga26/output/figurines_old/inpaint360gs/3dgs_output \
    --init_mode "sparse" \
    --eval \


export PYTHONPATH=$(pwd):$(pwd)/seg/detectron2:$PYTHONPATH
python seg/distillation.py \
    --source_path ../../siga26/data/figurines_old \
    --model_path ../../siga26/output/figurines_old/inpaint360gs \
    --vanilla_3dgs_path ../../siga26/output/figurines_old/inpaint360gs/3dgs_output \
    --object_path object_mask \
    --eval

python render.py -m ../../siga26/output/figurines_old/inpaint360gs --render_video