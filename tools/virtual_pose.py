# This file is part of inpaint360gs: Inpaint360GS: Efficient Object-Aware 3D Inpainting via Gaussian Splatting for 360° Scenes
# Project page: https://dfki-av.github.io/inpaint360gs/
#
# Copyright 2024-2026 Shaoxiang Wang <shaoxiang.wang@dfki.de>
# Licensed under the Apache License, Version 2.0.
# http://www.apache.org/licenses/LICENSE-2.0
#
# This file contains original research code and modified components from the 
# aforementioned projects. It is distributed on an "AS IS" BASIS, 
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. 
# See the License for the specific language governing permissions and 
# limitations under the License.

import argparse
import os
import sys
from pathlib import Path

import torch
from tqdm import tqdm
import numpy as np
from os import makedirs
from argparse import ArgumentParser
import cv2
import json
import copy

import torchvision
from PIL import Image

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scene import Scene
from gaussian_renderer import render
from utils.general_utils import safe_state, compose_camera_gt_with_background
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
from utils.graphics_utils import getWorld2View2
from utils.pose_utils import generate_ellipse_path, generate_virtual_radius
from render import render_set as render_set_full_scene_stage
# from edit_object_removal import render_set as render_set_removal_stage
from render import visualize_obj
from utils.point_utils import create_point_cloud, ply_color_fusion, get_intrinsics
from utils.iterative_workflow import prepare_manual_mask_request

def render_set_removal_stage(model_path, name, iteration, views, gaussians, pipeline, background, classifier, storage_mode="full"):
    """
    
    """
    print(f"\nIteration is {iteration}")
    iteration_step = iteration.split('_')[-1]
    render_path = os.path.join(model_path, name, "ours{}".format(iteration), "renders")
    gts_path = os.path.join(model_path, name, "ours{}".format(iteration), "gt")
    gt_colormask_path = os.path.join(model_path, name, "ours{}".format(iteration), "gt_objects_color")
    pred_obj_path = os.path.join(model_path, name, "ours{}".format(iteration), "objects_pred")
    pred_obj_color_path = os.path.join(model_path, name, "ours{}".format(iteration), "objects_pred_color")
    depth_path=os.path.join(model_path, name, "ours{}".format(iteration), "depth")               
    depth_original_path=os.path.join(model_path, name, "ours_{}".format(iteration_step), "depth")     
    
    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)
    makedirs(gt_colormask_path, exist_ok=True)
    makedirs(pred_obj_path, exist_ok=True)
    makedirs(pred_obj_color_path, exist_ok=True)
    makedirs(depth_path, exist_ok=True)

    save_fused_ply = storage_mode == "full"
    fused_vanilla_col_dep_ply_path=os.path.join(model_path, name, "ours{}".format(iteration), "fused_vanilla_col_dep_ply")
    if save_fused_ply:
        makedirs(fused_vanilla_col_dep_ply_path, exist_ok=True)

    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        results = render(view, gaussians, pipeline, background)
        rendering = results["render"]
        rendering_obj = results["render_object"]
        logits = classifier(rendering_obj)       
        pred_obj_mask = torch.argmax(logits,dim=0)
        pred_obj_color_mask = visualize_obj(pred_obj_mask.cpu().numpy().astype(np.uint8))
        gt_objects = view.objects
        gt_rgb_mask = visualize_obj(gt_objects.cpu().numpy().astype(np.uint8))
        depth=results["depth_3dgs"].squeeze(0).detach().cpu().numpy()
        np.save(os.path.join(depth_path, view.image_name+".npy"),depth)          

        depth_max = np.load(os.path.join(depth_original_path, view.image_name+".npy")).max()
        depth_min = np.load(os.path.join(depth_original_path, view.image_name+".npy")).min()
        depth = (depth - depth_min) / (depth_max - depth_min)             
        depth = (depth * 255.0).astype(np.uint8)
        depth = cv2.applyColorMap(depth, cv2.COLORMAP_JET)
        cv2.imwrite(os.path.join(depth_path, view.image_name + ".png"), depth)
        Image.fromarray(gt_rgb_mask).save(os.path.join(gt_colormask_path, view.image_name + ".png"))

        pred_obj_mask = pred_obj_mask.cpu().numpy().astype(np.uint8)
        Image.fromarray(pred_obj_mask).save(os.path.join(pred_obj_path, view.image_name + ".png"))
        Image.fromarray(pred_obj_color_mask).save(os.path.join(pred_obj_color_path, view.image_name + ".png"))
        gt = compose_camera_gt_with_background(view, background)
        torchvision.utils.save_image(rendering, os.path.join(render_path, view.image_name + ".png"))
        torchvision.utils.save_image(gt, os.path.join(gts_path, view.image_name + ".png"))

        if save_fused_ply:
            w2c = np.zeros((4, 4))
            w2c[:3, :3] = view.R.transpose()    # view.R: camera to world
            w2c[:3, 3] = view.T                 # view.T: world to camera
            w2c[3, 3] = 1.0
            c2w = np.linalg.inv(w2c)
            intrinsics = get_intrinsics(view.image_height, view.image_width,view.FoVx,view.FoVy)
            depth = np.load(os.path.join(depth_path, view.image_name +".npy"))
            points = create_point_cloud(depth, intrinsics, c2w)
            colors = cv2.imread(os.path.join(render_path, view.image_name + ".png")).reshape(-1,3)
            ply_path = os.path.join(fused_vanilla_col_dep_ply_path, view.image_name+".ply")
            ply_color_fusion(points, colors, ply_path)


def  virtual(dataset : ModelParams, iteration : int, pipeline : PipelineParams):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)
 
        target_object_physical_radius = args.target_object_radius

        classifier = torch.nn.Conv2d(gaussians.num_objects, dataset.num_classes, kernel_size=1) 
        classifier.cuda()
        classifier.load_state_dict(torch.load(os.path.join(dataset.model_path,"point_cloud","iteration_"+str(scene.loaded_iter),"classifier.pth")))

        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        views = scene.getTrainCameras()
        view = views[0]
        is_circle=True

        # here we calculate virtual camera pose circle radius
        if args.circle_radius != -1:
            pass
        else:
            args.circle_radius = generate_virtual_radius(views, 
                                    target_object_radius=target_object_physical_radius)

        poses = generate_ellipse_path(views, n_frames=30, 
                                    is_circle=is_circle, circle_radius=args.circle_radius)
        
        # save the circle radius ration in removal and inpaint config file
        config_paths = [args.config_file]
        if getattr(args, "inpaint_config_file", None):
            config_paths.append(args.inpaint_config_file)
        else:
            config_paths.append(args.config_file.replace("object_removal", "object_inpaint"))
        
        for path in config_paths:
            with open(path, "r") as f:
                scene_info = json.load(f)
            scene_info["circle_radius"] = round(args.circle_radius, 4)
            json_str = json.dumps(scene_info, indent=4, ensure_ascii=False)
            
            import re
            json_str = re.sub(
                r'\[\s+([\d, \s.-]+)\s+\]', 
                lambda m: "[" + re.sub(r'\s+', ' ', m.group(1).strip()) + "]", 
                json_str)
            
            with open(path, "w", encoding='utf-8') as f:
                f.write(json_str)

        virtual_pose_list = []
        for idx, pose in enumerate(tqdm(poses, desc="Prepare virtual camera pose")):
            view_tmp = copy.deepcopy(view)
            view_tmp.world_view_transform = torch.tensor(getWorld2View2(pose[:3, :3].T, pose[:3, 3], view.trans, view.scale)).transpose(0, 1).cuda()
            view_tmp.full_proj_transform = (view_tmp.world_view_transform.unsqueeze(0).bmm(view.projection_matrix.unsqueeze(0))).squeeze(0)
            view_tmp.camera_center = view_tmp.world_view_transform.inverse()[3, :3]
            view_tmp.image_name = f"{idx:05d}"
            view_tmp.R = pose[:3, :3].T
            view_tmp.T = pose[:3, 3]
            
            virtual_pose_list.append(view_tmp)

        # Step 1: Generate the full scene containing all objects
        render_set_full_scene_stage(dataset.model_path, "virtual", scene.loaded_iter, virtual_pose_list, gaussians, pipeline, background, classifier, storage_mode=args.storage_mode)

        # # Step 2: Generate the background scene with objects(target + surrounding) removed
        step_num = str(scene.loaded_iter)
        load_iteration='_object_removal/iteration_'+step_num
        scene = Scene(dataset, gaussians, load_iteration=load_iteration, shuffle=False) # load removal scene
        render_set_removal_stage(dataset.model_path, "virtual", load_iteration, virtual_pose_list, gaussians, pipeline, background, classifier, storage_mode=args.storage_mode)

        if args.mask_provider_type == "manual_sam2":
            renders_dir = os.path.join(args.model_path, "virtual", f"ours{scene.loaded_iter}", "renders")
            round_dir = args.round_dir if args.round_dir is not None else args.model_path
            request_manifest = prepare_manual_mask_request(
                round_dir,
                renders_dir,
                sync_segment_track_assets=args.sync_segment_track_assets,
                extra_manifest={
                    "virtual_iteration": scene.loaded_iter,
                    "model_path": args.model_path,
                },
            )
            print(f"\n📦 Manual mask-provider package prepared at: {request_manifest['request_zip_path']}")

        # Step3: Generate the background scene with objects(only target object) removed
        if len(args.select_obj_id) > 1 and len(args.surrounding_ids) > 0:
            scene = Scene(dataset, gaussians, load_iteration=f'_object_removal/iteration_{step_num}_removal_target', shuffle=False) # load removal scene
            render_set_full_scene_stage(dataset.model_path, "virtual", f'object_removal/iteration_{step_num}_removal_target', virtual_pose_list, gaussians, pipeline, background, classifier, storage_mode=args.storage_mode)
        
if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--skip_ellipse_video", action="store_true")
    parser.add_argument("--is_circle", action="store_false")
    parser.add_argument("--circle_radius", default=-1.0, type=float, help="smaller ratio means closer camera to object")
    parser.add_argument("--skip_gaussians_disturb", action="store_true")
    parser.add_argument("--mean", default=0, type=float)
    parser.add_argument("--std", default=0.03, type=float)
    parser.add_argument("--config_file", type=str, default="config/object_removal/inpaint360/doppelherz.json", help="Path to the configuration file")
    parser.add_argument("--inpaint_config_file", type=str, default=None, help="Optional paired inpaint configuration file updated with circle radius.")
    parser.add_argument("--round_dir", type=str, default=None, help="Optional iterative round directory used to store mask-provider artifacts.")
    parser.add_argument("--mask_provider_type", type=str, default="manual_sam2", help="Initial virtual-view mask provider. manual_sam2 keeps the current manual workflow; other values reserve the interface for future automation.")
    parser.add_argument("--sync_segment_track_assets", action=argparse.BooleanOptionalAction, default=True, help="Also mirror the manual mask package to Segment-and-Track-Anything/assets/images.zip.")
    parser.add_argument("--storage_mode", choices=["full", "lite", "minimal"], default="full", help="Output retention mode.")

    args = get_combined_args(parser)  

    with open(args.config_file, 'r') as file:
        config = json.load(file)

    args.select_obj_id = config.get("select_obj_id")
    args.surrounding_ids = config.get("surrounding_ids")
    args.target_object_radius = config.get("target_object_radius")

    # Initialize system state (RNG)
    safe_state(args.quiet)

    virtual(model.extract(args), args.iteration, pipeline.extract(args))

    # # python virtual_pose.py --source_path PATH/TO/DATASET --model_path PATH/TO/MODEL
