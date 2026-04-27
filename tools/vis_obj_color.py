import os
import numpy as np
from PIL import Image
from render import visualize_obj

def vis_mask_images(input_folder: str, output_folder: str):
    
    os.makedirs(output_folder, exist_ok=True)
    print(f"Saved our visualized image under : {output_folder}")
    
    for image_name in os.listdir(input_folder):
        if not image_name.endswith('.png'):
            continue

        file_path = os.path.join(input_folder, image_name)
        
        pred_obj_mask = np.array(Image.open(file_path), dtype=np.uint8)
        # print(f"Processing {file_path}: original shape = {pred_obj_mask.shape}")

        if len(pred_obj_mask.shape) == 3 and pred_obj_mask.shape[-1] == 3:
            print(f"Warning: {image_name} is RGB, extracting the first channel...")
            pred_obj_mask = pred_obj_mask[:, :, 0]
        pred_obj_mask = pred_obj_mask.squeeze()
        
        pred_obj_color_mask = visualize_obj(pred_obj_mask).astype(np.uint8)
        
        save_path = os.path.join(output_folder, image_name.replace(".png", ".png"))
        # save_path = os.path.join(output_folder, image_name.replace(".png", "_color.png"))
        Image.fromarray(pred_obj_color_mask).save(save_path)
