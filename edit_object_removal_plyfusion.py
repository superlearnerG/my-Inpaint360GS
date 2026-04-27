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

import torch
from scene import Scene
import os
from pathlib import Path
from tqdm import tqdm
from os import makedirs
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, OptimizationParams, get_combined_args
from utils.point_utils import create_point_cloud, ply_color_fusion, get_intrinsics
from gaussian_renderer import GaussianModel
import numpy as np
import json
import cv2
from utils.pose_utils import generate_ellipse_path
from utils.graphics_utils import getWorld2View2
import copy
from utils.inpaint_target_paths import (
    find_image_for_stem,
    get_ready_for_3dinpaint_color_dir,
    get_ready_for_3dinpaint_depth_completed_dir,
    get_unseen_mask_ready_dir,
)
from utils.iterative_workflow import write_json


def normalize_support_view_name(support_view_name):
    if support_view_name is None:
        return None
    stem = Path(str(support_view_name)).stem
    if stem.isdigit():
        return f"{int(stem):05d}"
    return stem


def fusion(dataset_path, model_path, target_id, name, iteration, views, storage_mode="full", support_view_name=None):
    """
    
    """
    render_path = os.path.join(model_path, name, "ours_object_removal/iteration_{}".format(iteration), "renders")
    depth_hole_path=os.path.join(model_path, name, "ours_object_removal/iteration_{}".format(iteration), "depth")                    # result after removal
    fused_mask_col_dep_ply_path=os.path.join(model_path, name, "ours_object_removal/iteration_{}".format(iteration), "fused_mask_col_dep_ply")      
    fused_hole_col_dep_ply_path=os.path.join(model_path, name, "ours_object_removal/iteration_{}".format(iteration), "fused_hole_col_dep_ply")   
    ready_color_dir = get_ready_for_3dinpaint_color_dir(model_path, target_id)
    unseen_mask_ready_dir = get_unseen_mask_ready_dir(model_path, target_id)
    depth_completed_dir = get_ready_for_3dinpaint_depth_completed_dir(model_path, target_id)

    save_all_fused_ply = storage_mode == "full"
    makedirs(fused_mask_col_dep_ply_path, exist_ok=True)
    if save_all_fused_ply:
        makedirs(fused_hole_col_dep_ply_path, exist_ok=True)

    view = views[0]
    is_circle=True
    poses = generate_ellipse_path(views, n_frames=30, is_circle=is_circle, circle_radius=args.circle_radius)
    virtual_poses_list = []
    for idx, pose in enumerate(tqdm(poses, desc="Prepare virtual camera poses")):
        view_tmp = copy.deepcopy(view)
        view_tmp.world_view_transform = torch.tensor(getWorld2View2(pose[:3, :3].T, pose[:3, 3], view.trans, view.scale)).transpose(0, 1).cuda()
        view_tmp.full_proj_transform = (view_tmp.world_view_transform.unsqueeze(0).bmm(view.projection_matrix.unsqueeze(0))).squeeze(0)
        view_tmp.camera_center = view_tmp.world_view_transform.inverse()[3, :3]
        view_tmp.image_name = f"{idx:05d}"

        view_tmp.R = pose[:3, :3].T
        view_tmp.T = pose[:3, 3]
        
        virtual_poses_list.append(view_tmp)

    support_stem = normalize_support_view_name(support_view_name)
    if not save_all_fused_ply:
        if support_stem is None:
            support_stem = virtual_poses_list[len(virtual_poses_list) // 2].image_name
        valid_stems = {view.image_name for view in virtual_poses_list}
        if support_stem not in valid_stems:
            raise ValueError(f"--support_view_name must refer to one of {sorted(valid_stems)}, got {support_view_name}")

    fused_mask_ply_paths = []
    fused_hole_ply_paths = []

    for idx, view in enumerate(tqdm(virtual_poses_list, desc="Color-Depth-Fusion progress")):
        if not save_all_fused_ply and view.image_name != support_stem:
            continue

        w2c = np.zeros((4, 4))
        w2c[:3, :3] = view.R.transpose()    # view.R: camera to world
        w2c[:3, 3] = view.T                 # view.T: world to camera
        w2c[3, 3] = 1.0       
        c2w = np.linalg.inv(w2c)    
        intrinsics = get_intrinsics(view.image_height, view.image_width,view.FoVx,view.FoVy)

        inpainted_2d_color_path = find_image_for_stem(ready_color_dir, view.image_name)
        colors = cv2.imread(str(inpainted_2d_color_path)).reshape(-1,3)
        mask = cv2.imread(
            str(unseen_mask_ready_dir / f"{view.image_name}.png"),
            cv2.IMREAD_GRAYSCALE,
        ).astype(bool).reshape(-1)
        depth_completed = np.load(depth_completed_dir / f"{view.image_name}.npy")  
        
        points = create_point_cloud(depth_completed, intrinsics, c2w)
        ply_path = os.path.join(fused_mask_col_dep_ply_path, view.image_name+".ply")
        ply_color_fusion(points, colors, ply_path, mask=mask)
        fused_mask_ply_paths.append(ply_path)

        if save_all_fused_ply:
            # removal scene
            colors_hole = cv2.imread(os.path.join(render_path,  view.image_name+".png")).reshape(-1,3)
            depth_hole = np.load(os.path.join(depth_hole_path, view.image_name+".npy")) 
            points_hole = create_point_cloud(depth_hole, intrinsics, c2w)
            ply_hole_path = os.path.join(fused_hole_col_dep_ply_path, view.image_name+".ply")
            ply_color_fusion(points_hole, colors_hole, ply_hole_path)
            fused_hole_ply_paths.append(ply_hole_path)

    default_support_ply = None
    if fused_mask_ply_paths:
        default_support_ply = fused_mask_ply_paths[len(fused_mask_ply_paths) // 2]
    else:
        raise RuntimeError("No support PLY was generated for 3D inpaint.")

    manifest = {
        "target_id": target_id,
        "iteration": iteration,
        "storage_mode": storage_mode,
        "fused_mask_col_dep_ply_dir": fused_mask_col_dep_ply_path,
        "fused_hole_col_dep_ply_dir": fused_hole_col_dep_ply_path,
        "fused_mask_col_dep_ply_files": fused_mask_ply_paths,
        "fused_hole_col_dep_ply_files": fused_hole_ply_paths,
        "default_support_ply": default_support_ply,
    }
    manifest_path = os.path.join(model_path, name, f"ours_object_removal/iteration_{iteration}", "fusion_manifest.json")
    write_json(manifest_path, manifest)


def removal(dataset : ModelParams, iteration : int, pipeline : PipelineParams):
    
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)
   
    with torch.no_grad():
        fusion(
            dataset.source_path,
            dataset.model_path,
            args.target_id,
            "virtual",
            scene.loaded_iter,
            scene.getTrainCameras(),
            storage_mode=args.storage_mode,
            support_view_name=getattr(args, "support_view_name", None),
        )


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    opt = OptimizationParams(parser)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--quiet", action="store_true")

    parser.add_argument("--config_file", type=str, default="config/object_removal/inpaint360/picnic.json", help="Path to the configuration file")
    parser.add_argument("--storage_mode", choices=["full", "lite", "minimal"], default="full", help="Output retention mode.")
    parser.add_argument("--support_view_name", type=str, default=None, help="Optional virtual-view stem used to select the single support PLY in lite/minimal mode.")
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    # Read and parse the configuration file

    with open(args.config_file, 'r') as file:
        config = json.load(file)

    args.select_obj_id = config.get("select_obj_id")
    args.circle_radius = config.get("circle_radius")
    args.target_id = config.get("target_id")

    # Initialize system state (RNG)
    safe_state(args.quiet)

    removal(model.extract(args), args.iteration, pipeline.extract(args))
