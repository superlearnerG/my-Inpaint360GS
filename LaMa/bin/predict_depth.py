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
from pathlib import Path

from saicinpainting.evaluation.utils import move_to_device
from saicinpainting.evaluation.refinement import refine_predict

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.pretrained_paths import configure_pretrained_env, require_external_lama_dir

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
    configure_pretrained_env(include_simple_lama=False)

    default_config = OmegaConf.load('./configs/prediction/default.yaml')

    data_name = args.data_name
    indir = f'./data/depth/{data_name}'
    outdir = f'./output/depth/{data_name}'

    default_config = OmegaConf.load('./configs/prediction/default.yaml')
    default_config.dataset.img_suffix = '.npy' 
    

    custom_config = OmegaConf.create({
        'refine': True,
        'model': {
            'path': str(require_external_lama_dir())
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


        for img_i in tqdm.trange(len(dataset)):
            mask_fname = dataset.mask_filenames[img_i]
            os.makedirs(os.path.join(predict_config.outdir, "vis"), exist_ok=True)

            cur_out_fname = os.path.join(
                predict_config.outdir, "vis",
                os.path.splitext(mask_fname[len(predict_config.indir):])[0][:-5] + out_ext
            )

            os.makedirs(os.path.dirname(cur_out_fname), exist_ok=True)

            batch = default_collate([dataset[img_i]])
            if predict_config.get('refine', False):
                assert 'unpad_to_size' in batch, "Unpadded size is required for the refinement"
                # image unpadding is taken care of in the refiner, so that output image
                # is same size as the input image
                cur_res = refine_predict(batch, model, **predict_config.refiner)
                cur_res = cur_res[0].permute(1,2,0).detach().cpu().numpy()

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

            if "npy" in default_config.dataset.img_suffix:
                depth_original_path = os.path.join(predict_config.indir, "depth_original", os.path.splitext(mask_fname[len(predict_config.indir):])[0][:-5]+".npy")    # depth

                depth_max = np.load(depth_original_path).max()
                depth_min = np.load(depth_original_path).min()

                depth_completed = cur_res * (depth_max - depth_min) + depth_min
                depth_npy_path = os.path.join(
                                        predict_config.outdir, 
                                        os.path.splitext(mask_fname[len(predict_config.indir):])[0][:-5]+".npy")
                np.save(depth_npy_path, depth_completed[:, :, 0])

                depth_jet = (cur_res[:,:, 0] * 255.0).astype(np.uint8)      # single channel
                depth_jet = cv2.applyColorMap(depth_jet, cv2.COLORMAP_JET)  # three channel

                depth_jet_path = os.path.join(
                                        predict_config.outdir, "vis",
                                        os.path.splitext(mask_fname[len(predict_config.indir):])[0][:-5]+"_jet.png")
                cv2.imwrite(depth_jet_path, depth_jet)


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
    args = parser.parse_args()

    main(args)

# python bin/predict_depth.py --data_name 360_doppelherz_virtual
