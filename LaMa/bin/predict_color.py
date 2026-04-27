# This file is part of inpaint360gs: Inpaint360GS: Efficient Object-Aware 3D Inpainting via Gaussian Splatting for 360° Scenes
# Project page: https://dfki-av.github.io/inpaint360gs/
#
# Copyright 2024-2026 Shaoxiang Wang <shaoxiang.wang@dfki.de>
# Licensed under the Apache License, Version 2.0.
# http://www.apache.org/licenses/LICENSE-2.0

# Modified from codes in LaMa https://github.com/advimman/lama

# This file contains original research code and modified components from the 
# aforementioned projects. It is distributed on an "AS IS" BASIS, 
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. 
# See the License for the specific language governing permissions and 
# limitations under the License.

#!/usr/bin/env python3

import logging
import os
import sys
import traceback
import pdb
from saicinpainting.evaluation.utils import move_to_device
from saicinpainting.evaluation.refinement import refine_predict

os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'


import cv2
import hydra
import numpy as np
import torch
import tqdm
import yaml
import argparse
from omegaconf import OmegaConf
from torch.utils.data._utils.collate import default_collate

from saicinpainting.training.data.datasets import make_default_val_dataset
from saicinpainting.training.trainers import load_checkpoint
from saicinpainting.utils import register_debug_signal_handlers

LOGGER = logging.getLogger(__name__)


def main(args):

    data_name = args.data_name
    indir = f'./data/color/{data_name}'
    outdir = f'./output/color/{data_name}'

    default_config = OmegaConf.load('./configs/prediction/default.yaml')
    default_config.dataset.img_suffix = '.png' 
    
    custom_config = OmegaConf.create({
        'refine': True,
        'model': {
            'path': './big-lama'        
        },
        'indir': indir,
        'outdir': outdir
    })
    
    predict_config = OmegaConf.merge(default_config, custom_config)

    print(predict_config)  

    try:
        if sys.platform != 'win32':
            register_debug_signal_handlers()  # kill -10 <pid> will result in traceback dumped into log

        device = torch.device("cpu")

        train_config_path = os.path.join(predict_config.model.path, 'config.yaml')

        with open(train_config_path, 'r') as f:
            train_config = OmegaConf.create(yaml.safe_load(f))
        
        train_config.training_model.predict_only = True
        train_config.visualizer.kind = 'noop'

        out_ext = predict_config.get('out_ext', '.png')

        checkpoint_path = os.path.join(predict_config.model.path, 
                                       'models', 
                                       predict_config.model.checkpoint)
        model = load_checkpoint(train_config, checkpoint_path, strict=False, map_location='cpu')
        model.freeze()
        if not predict_config.get('refine', False):
            model.to(device)

        if not predict_config.indir.endswith('/'):
            predict_config.indir += '/'
        
        dataset = make_default_val_dataset(predict_config.indir, **predict_config.dataset)

        if args.recursive_guide:
            # Note: Recursive guide is experimental and may not suit all scenes. Suggest to disable it in default."
            prev_out_fname = None
            for img_i in tqdm.trange(len(dataset)):
                img_fname = dataset.img_filenames[img_i]
                mask_fname = dataset.mask_filenames[img_i]
                
                cur_out_fname = os.path.join(
                    predict_config.outdir,
                    os.path.splitext(mask_fname[len(predict_config.indir):])[0][:-5] + out_ext
                )
                os.makedirs(os.path.dirname(cur_out_fname), exist_ok=True)

                item = dataset[img_i]
                curr_img = item['image']  # Tensor (3, H, W)
                curr_mask = item['mask']   # Tensor (1, H, W)

                if isinstance(curr_img, np.ndarray):
                    curr_img = torch.from_numpy(curr_img).float()
                if isinstance(curr_mask, np.ndarray):
                    curr_mask = torch.from_numpy(curr_mask).float()

                if img_i > 0 and prev_out_fname is not None and os.path.exists(prev_out_fname):
                    prev_res_bgr = cv2.imread(prev_out_fname)
                    prev_res_rgb = cv2.cvtColor(prev_res_bgr, cv2.COLOR_BGR2RGB)
                    
                    prev_res_tensor = torch.from_numpy(prev_res_rgb).permute(2, 0, 1).float() / 255.0
                    
                    curr_c, curr_h, curr_w = curr_img.shape
                    prev_c, prev_h, prev_w = prev_res_tensor.shape

                    if prev_h != curr_h or prev_w != curr_w:
                
                        prev_res_tensor = torch.nn.functional.interpolate(
                            prev_res_tensor.unsqueeze(0), 
                            size=(curr_h, curr_w), 
                            mode='bilinear', 
                            align_corners=False
                        ).squeeze(0)

                    combined_img = torch.cat([prev_res_tensor, curr_img], dim=2)
                    prev_mask_blank = torch.zeros_like(curr_mask)
                    combined_mask = torch.cat([prev_mask_blank, curr_mask], dim=2)
                    
                    batch_item = item.copy()
                    batch_item['image'] = combined_img
                    batch_item['mask'] = combined_mask
                    
                    if 'unpad_to_size' in batch_item:
                        h, w = batch_item['unpad_to_size']
                        batch_item['unpad_to_size'] = (curr_h, curr_w * 2)
                else:

                    batch_item = item.copy()
                    batch_item['image'] = curr_img
                    batch_item['mask'] = curr_mask


                batch = default_collate([batch_item])
                
                if predict_config.get('refine', False):
                    cur_res = refine_predict(batch, model, **predict_config.refiner)
                    cur_res = cur_res[0].permute(1, 2, 0).detach().cpu().numpy()
                else:
                    with torch.no_grad():
                        batch = move_to_device(batch, device)
                        batch['mask'] = (batch['mask'] > 0) * 1
                        batch = model(batch)
                        cur_res = batch[predict_config.out_key][0].permute(1, 2, 0).detach().cpu().numpy()
                        
                # --- Cropping results and high-resolution local restoration ---
                if img_i > 0:
                    h_low, total_w_low, _ = cur_res.shape
                    cur_res = cur_res[:, (total_w_low // 2):, :]

                orig_img_bgr = cv2.imread(img_fname)
                orig_mask_bgr = cv2.imread(mask_fname)
                orig_h, orig_w = orig_img_bgr.shape[:2]

                inpainted_rgb = np.clip(cur_res * 255, 0, 255).astype('uint8')
                inpainted_bgr = cv2.cvtColor(inpainted_rgb, cv2.COLOR_RGB2BGR)

                inpainted_full_bgr = cv2.resize(inpainted_bgr, (orig_w, orig_h), interpolation=cv2.INTER_LANCZOS4)
                mask_binary = (orig_mask_bgr > 127).astype(np.uint8) 
                final_res_bgr = orig_img_bgr * (1 - mask_binary) + inpainted_full_bgr * mask_binary

                cv2.imwrite(cur_out_fname, final_res_bgr)
                
                prev_out_fname = cur_out_fname
        
        else:
            for img_i in tqdm.trange(len(dataset)):
                mask_fname = dataset.mask_filenames[img_i]

                cur_out_fname = os.path.join(
                    predict_config.outdir,
                    os.path.splitext(mask_fname[len(predict_config.indir):])[0][:-5] + out_ext
                )

                os.makedirs(os.path.dirname(cur_out_fname), exist_ok=True)

                batch = default_collate([dataset[img_i]])
                if predict_config.get('refine', False):
                    assert 'unpad_to_size' in batch, "Unpadded size is required for the refinement"
                    # image unpadding is taken care of in the refiner, so that output image
                    # is same size as the input image
                    cur_res = refine_predict(batch, model, **predict_config.refiner)
                    cur_res = cur_res[0].permute(1,2,0).detach().cpu().numpy() # here is inpainted image！   
                else:
                    with torch.no_grad():
                        batch = move_to_device(batch, device)
                        batch['mask'] = (batch['mask'] > 0) * 1
                        batch = model(batch)                    
                        cur_res = batch[predict_config.out_key][0].permute(1, 2, 0).detach().cpu().numpy()
                        unpad_to_size = batch.get('unpad_to_size', None)
                        if unpad_to_size is not None:
                            orig_height, orig_width = unpad_to_size
                            cur_res = cur_res[:orig_height, :orig_width]

                cur_res = np.clip(cur_res * 255, 0, 255).astype('uint8')  

                cur_res = cv2.cvtColor(cur_res, cv2.COLOR_RGB2BGR)
                cv2.imwrite(cur_out_fname, cur_res)
                # print(f"The currrent output is saved at {cur_out_fname}.")

    except KeyboardInterrupt:
        LOGGER.warning('Interrupted by user')
    except Exception as ex:
        LOGGER.critical(f'Prediction failed due to {ex}:\n{traceback.format_exc()}')
        sys.exit(1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run LaMa inpainting prediction")
    parser.add_argument("--data_name", type=str, required=True, help="e.g. 360_doppelherz")
    parser.add_argument("--recursive_guide", action='store_true', help="Enable recursive guidance")
    args = parser.parse_args()

    main(args)

# python bin/predict_color.py --data_name 360_fruits_virtual
