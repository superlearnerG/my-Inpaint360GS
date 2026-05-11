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

import numpy as np
import open3d as o3d
from scene import Scene
from plyfile import PlyData, PlyElement
import torch
import os
from os import makedirs, path
from errno import EEXIST
from sklearn.neighbors import KDTree
from gaussian_renderer import render
from sklearn.cluster import DBSCAN
from tqdm import tqdm
from simple_knn._C import distCUDA2

import lpips
from random import randint
from torch import nn
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams, get_combined_args
from gaussian_renderer import GaussianModel
import json
from tqdm import tqdm
from render import visualize_obj, render_video_func_wriva
from utils.loss_utils import masked_l1_loss, ssim, masked_ssim
from PIL import Image
import torchvision
import cv2
from edit_object_removal import points_inside_convex_hull
from utils.general_utils import safe_state, compose_camera_gt_with_background
from utils.pose_utils import generate_ellipse_path
from utils.graphics_utils import getWorld2View2,getProjectionMatrix
from utils.general_utils import PILtoTorch
import copy
from utils.point_utils import project_3d_points,ndc_to_pixel
from utils.inpaint_target_paths import (
    find_image_for_stem,
    get_ready_for_3dinpaint_color_dir,
    get_unseen_mask_ready_dir,
)
from utils.iterative_workflow import resolve_support_ply
from utils.pretrained_paths import configure_pretrained_env

C0 = 0.28209479177387814
max_sh_degree = 3            

def mask_to_bbox(mask):
    # Find the rows and columns where the mask is non-zero
    rows = torch.any(mask, dim=1)
    cols = torch.any(mask, dim=0)
    ymin, ymax = torch.where(rows)[0][[0, -1]]
    xmin, xmax = torch.where(cols)[0][[0, -1]]
    
    return xmin, ymin, xmax, ymax

def crop_using_bbox(image, bbox):
    xmin, ymin, xmax, ymax = bbox
    return image[:, ymin:ymax+1, xmin:xmax+1]

# Function to divide image into K x K patches
def divide_into_patches(image, K):
    B, C, H, W = image.shape
    patch_h, patch_w = H // K, W // K
    patches = torch.nn.functional.unfold(image, (patch_h, patch_w), stride=(patch_h, patch_w))
    patches = patches.view(B, C, patch_h, patch_w, -1)    
    return patches.permute(0, 4, 1, 2, 3)

def construct_list_of_attributes(features_dc,features_rest,scaling,rotation, objects_dc):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        # All channels except the 3 DC
        for i in range(features_dc.shape[1]*features_dc.shape[2]):
            l.append('f_dc_{}'.format(i))
        for i in range(features_rest.shape[1]*features_rest.shape[2]):
            l.append('f_rest_{}'.format(i))
        l.append('opacity')
        for i in range(scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(rotation.shape[1]):
            l.append('rot_{}'.format(i))
        for i in range(objects_dc.shape[1]*objects_dc.shape[2]):
            l.append('obj_dc_{}'.format(i))
        return l


def mkdir_p(folder_path):
    # Creates a directory. equivalent to using mkdir -p on the command line
    try:
        makedirs(folder_path)
    except OSError as exc: 
        if exc.errno == EEXIST and path.isdir(folder_path):
            pass
        else:
            raise


def save_ply(xyz, features_dc, features_rest, opacity, scaling, rotation, objects_dc, path_save):
    """
    
    """
    mkdir_p(os.path.dirname(path_save))

    xyz = xyz.detach().cpu().numpy()
    normals = np.zeros_like(xyz)
    f_dc = features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
    f_rest = features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
    opacities = opacity.detach().cpu().numpy()
    scale = scaling.detach().cpu().numpy()
    rotation = rotation.detach().cpu().numpy()
    obj_dc = objects_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()

    dtype_full = [(attribute, 'f4') for attribute in construct_list_of_attributes(features_dc, features_rest, scaling, rotation, objects_dc)]

    elements = np.empty(xyz.shape[0], dtype=dtype_full)
    attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation, obj_dc), axis=1)
    elements[:] = list(map(tuple, attributes))
    el = PlyElement.describe(elements, 'vertex')
    PlyData([el]).write(path_save)
    print("The new point cloud are saved at {}".format(path_save))


def get_projected_gaussians(gaussians, viewpoint, supp_ply_path=None): 
    """
    Project 3D Gaussian points to the 2D image plane and filter out points
    that fall outside the image bounds.

    Return:
        p_inside_mask: mask for points inside the image
        p_inside_obj_mask: mask for points inside the object region
    """
    proj_matrix = viewpoint.full_proj_transform

    W = viewpoint.image_width
    H = viewpoint.image_height

    obj_mask = (viewpoint.objects.detach() > 0).to(torch.uint8)
    obj_mask_np = obj_mask.cpu().numpy()
    original_area = np.sum(obj_mask_np)
    target_area = int(original_area * 1.10)
    for k in range(3, 101, 2):
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        dilated = cv2.dilate(obj_mask_np, kernel)
        if np.sum(dilated) >= target_area:
            break 
    obj_mask = torch.from_numpy(dilated).to(device=viewpoint.objects.device).bool()

    p_hom = project_3d_points(gaussians.get_xyz, proj_matrix)  # (N, 4)
    p_hom_z = p_hom[:, 2]

    p_w = 1 / (p_hom[:, 3:] + 1e-8)
    p_proj = p_hom[:, :3] * p_w
    p_proj[:, 0] = ndc_to_pixel(p_proj[:, 0], W)
    p_proj[:, 1] = ndc_to_pixel(p_proj[:, 1], H)
    p_proj = torch.round(p_proj[:, :2]).long()

    p_inside_mask = (p_proj[:, 0] >= 0) & (p_proj[:, 0] < W) & (p_proj[:, 1] >= 0) & (p_proj[:, 1] < H) & (p_hom_z > 0)

    p_proj_inside = p_proj[p_inside_mask]  # (M, 2)
    x_coords, y_coords = p_proj_inside[:, 0], p_proj_inside[:, 1]
    obj_mask_values = obj_mask[y_coords, x_coords]
    
    p_inside_obj_mask = torch.zeros_like(p_inside_mask)    
    p_inside_obj_mask[p_inside_mask] = obj_mask_values

    # --- Spatial Depth Filter ---
    if os.path.exists(args.supp_ply):
        from scipy.spatial import cKDTree
        
        # 1. Load seed points and build index
        supp_ply = PlyData.read(args.supp_ply)
        supp_xyz = np.stack([np.asarray(supp_ply.elements[0][axis]) for axis in 'xyz'], axis=1)
        tree = cKDTree(supp_xyz)

        # 2. Define adaptive threshold based on seed distribution (e.g., 3x average STD)
        adaptive_threshold = np.std(supp_xyz, axis=0).mean() * 3.0

        # 3. Query distances for 2D-masked candidates only
        candidate_indices = torch.where(p_inside_obj_mask)[0]
        if len(candidate_indices) > 0:
            candidate_xyz = gaussians.get_xyz[candidate_indices].detach().cpu().numpy()
            dists, _ = tree.query(candidate_xyz)

            # 4. Refine mask by spatial proximity
            valid_mask = dists < adaptive_threshold
            final_mask = torch.zeros_like(p_inside_obj_mask, dtype=torch.bool)
            final_mask[candidate_indices[valid_mask]] = True
            p_inside_obj_mask = final_mask
   
    p_proj_inside = p_proj[p_inside_mask]
    projected_gaussian = {
        "p_inside_mask": p_inside_mask,       
        "p_inside_obj_mask": p_inside_obj_mask, 
    }

    return projected_gaussian


def finetune_inpaint(args, opt, model_path, iteration, views, gaussians, pipeline, background, classifier, selected_obj_ids, cameras_extent, removal_thresh, finetune_iteration):

    removal_iteration = iteration
    selected_obj_ids = torch.tensor(selected_obj_ids).cuda()
    masks_per_obj = dict()

    # get 3d gaussians idx corresponding to select obj id
    with torch.no_grad():
        if max(selected_obj_ids) >= 256:
            mask3d = torch.zeros_like(gaussians._xyz[:, 0], dtype=torch.bool, device="cuda")
        else:
            logits3d = classifier(gaussians._objects_dc.permute(2,0,1))
            prob_obj3d = torch.softmax(logits3d,dim=0)

            for obj_id in selected_obj_ids:
                obj_id_int = int(obj_id.item())
                obj_prob = prob_obj3d[obj_id, :, :]
                mask = obj_prob > removal_thresh
                mask3d = mask.squeeze()

                mask3d_convex, _ = points_inside_convex_hull(
                    gaussians._xyz.detach(), mask3d, remove_outliers=True, outlier_factor=1.0
                )
                mask3d = torch.logical_or(mask3d,mask3d_convex)
                mask3d = mask3d.float()[:,None,None]

                masks_per_obj[obj_id_int] = {
                "mask":  mask.float()[:,None],
                "mask3d": mask3d
                }

    # initialize gaussians
    gaussians.inpaint_setup(args, opt, masks_per_obj)

    removal_gaussian = GaussianModel(gaussians.max_sh_degree)
    removal_gaussian._xyz           = nn.Parameter(gaussians._xyz[:gaussians.sub_feature_num].detach().clone())
    removal_gaussian._features_dc   = nn.Parameter(gaussians._features_dc[:gaussians.sub_feature_num].detach().clone())
    removal_gaussian._features_rest = nn.Parameter(gaussians._features_rest[:gaussians.sub_feature_num].detach().clone())
    removal_gaussian._opacity       = nn.Parameter(gaussians._opacity[:gaussians.sub_feature_num].detach().clone())
    removal_gaussian._scaling       = nn.Parameter(gaussians._scaling[:gaussians.sub_feature_num].detach().clone())
    removal_gaussian._rotation      = nn.Parameter(gaussians._rotation[:gaussians.sub_feature_num].detach().clone())
    removal_gaussian._objects_dc    = nn.Parameter(gaussians._objects_dc[:gaussians.sub_feature_num].detach().clone())

    iterations = finetune_iteration    
    progress_bar = tqdm(range(iterations), desc="Finetuning progress")
    configure_pretrained_env(include_simple_lama=False)
    LPIPS = lpips.LPIPS(net='vgg')
    for param in LPIPS.parameters():
        param.requires_grad = False      
    LPIPS.cuda()

    valid_views = [view for view in views if torch.any(view.objects > 128).item()]
    skipped_view_count = len(views) - len(valid_views)
    if skipped_view_count > 0:
        print(f"\nSkip {skipped_view_count} virtual views with empty unseen masks during finetuning.")
    if not valid_views:
        raise RuntimeError("No virtual views with non-empty unseen masks found for inpaint finetuning.")

    for train_iter in range(iterations):
        viewpoint_stack = valid_views.copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))
        render_pkg = render(viewpoint_cam, gaussians, pipeline, background)
        image, viewspace_point_tensor, visibility_filter, radii, objects = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"], render_pkg["render_object"]

        mask2d = viewpoint_cam.objects > 128
        if not torch.any(mask2d):
            continue
        gt_image = compose_camera_gt_with_background(viewpoint_cam, background).cuda()
        Ll1 = masked_l1_loss(image, gt_image, ~mask2d)  

        bbox = mask_to_bbox(mask2d)
        cropped_image = crop_using_bbox(image, bbox)
        cropped_gt_image = crop_using_bbox(gt_image, bbox)
        K = 2
        rendering_patches = divide_into_patches(cropped_image[None, ...], K)  
        gt_patches = divide_into_patches(cropped_gt_image[None, ...], K)
        
        if rendering_patches.shape[-2] >= 32 and rendering_patches.shape[-1] >= 32:
            lpips_loss = LPIPS(rendering_patches.squeeze()*2 - 1, gt_patches.squeeze()*2 - 1).mean()
        else:
            lpips_loss = torch.tensor(0.0, device=rendering_patches.device)
       
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))  + args.lambda_lpips * lpips_loss

        loss.backward()

        with torch.no_grad():
            if train_iter < 5000 :
                # Keep track of max radii in image-space for pruning
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if  train_iter > 500 and train_iter % 100 == 0 :
                    size_threshold = 20
                    gaussians.densify_and_prune_inpaint(opt.densify_grad_threshold, 0.005, cameras_extent, size_threshold, gaussians.sub_feature_num)
                
        gaussians.optimizer.step()
        gaussians.optimizer.zero_grad(set_to_none = True)

        if train_iter % 10 == 0:
            progress_bar.set_postfix({"Loss": f"{loss:.{7}f}"})
            progress_bar.update(10)
    progress_bar.close()

    with torch.no_grad():
        tmp_gaussians = copy.deepcopy(gaussians) 
        projected_gaussian = get_projected_gaussians(tmp_gaussians, views[0], supp_ply_path=args.supp_ply )   
        p_inside_obj_mask = projected_gaussian["p_inside_obj_mask"]

        gaussians._xyz[:gaussians.sub_feature_num]           = removal_gaussian._xyz
        gaussians._features_dc[:gaussians.sub_feature_num]   = removal_gaussian._features_dc
        gaussians._features_rest[:gaussians.sub_feature_num] = removal_gaussian._features_rest
        gaussians._opacity[:gaussians.sub_feature_num]       = removal_gaussian._opacity
        gaussians._scaling[:gaussians.sub_feature_num]       = removal_gaussian._scaling 
        gaussians._rotation[:gaussians.sub_feature_num]      = removal_gaussian._rotation
        gaussians._objects_dc[:gaussians.sub_feature_num]    = removal_gaussian._objects_dc

        fields_to_update = [
        "_xyz", "_features_dc", "_features_rest", "_opacity", "_scaling", "_rotation", "_objects_dc"
        ]

        for field in fields_to_update:
            getattr(gaussians, field)[p_inside_obj_mask] = getattr(tmp_gaussians, field)[p_inside_obj_mask]

    # save gaussians
    point_cloud_path = os.path.join(model_path, "point_cloud_object_inpaint_virtual", "iteration_{}".format(iterations))
    gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))

    # recover surrounding objects back
    if len(args.surrounding_ids) > 0 and len(args.target_id) < len(args.select_obj_id):
        print(f"\nCombine objects{args.surrounding_ids} back.")
        from tools.combine_gaussian_scene import combine_gaussian
        gaussians = combine_gaussian(model.extract(args), 
                        f"_object_inpaint_virtual/iteration_{iterations}/point_cloud.ply",
                        pipeline, args.surrounding_ids, removal_iteration=removal_iteration)

    return gaussians

def render_set(model_path, name, iteration, views, gaussians, pipeline, background, classifier, args):
    """
    Args:
        name: "test" or "train" or "inpaint"

    """
    print(f"\nIteration is {iteration}")
    save_folder = os.path.join(model_path, name, "ours{}".format(iteration))

    render_path = os.path.join(save_folder, "renders")
    gts_path = os.path.join(save_folder, "gt")
    depth_path=os.path.join(save_folder, "depth")
    depth_original_path=os.path.join(model_path, name, f"ours{iteration}", "depth")

    makedirs(save_folder, exist_ok=True)
    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)
    makedirs(depth_path, exist_ok=True)

    with open(os.path.join(save_folder, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        results = render(view, gaussians, pipeline, background)
        rendering = results["render"]
        rendering_obj = results["render_object"]
        logits = classifier(rendering_obj)
        pred_obj_mask = torch.argmax(logits,dim=0)
        pred_obj_color_mask = visualize_obj(pred_obj_mask.cpu().numpy().astype(np.uint8))

        gt_objects = view.objects

        if gt_objects == None:
            pass
        else:
            gt_rgb_mask = visualize_obj(gt_objects.cpu().numpy().astype(np.uint8))
      
        depth=results["depth_3dgs"].squeeze(0).detach().cpu().numpy()
        np.save(os.path.join(depth_path, view.image_name+".npy"),depth)
        
        if name=="inpaint":
            depth_max = depth.max()
            depth_min = depth.min()
        else:
            depth_max = np.load(os.path.join(depth_original_path, view.image_name+".npy")).max()
            depth_min = np.load(os.path.join(depth_original_path, view.image_name+".npy")).min()

        depth = (depth - depth_min) / (depth_max - depth_min)
        depth = (depth * 255.0).astype(np.uint8)
        depth = cv2.applyColorMap(depth, cv2.COLORMAP_JET)
        cv2.imwrite(os.path.join(depth_path, view.image_name + ".png"), depth)

        pred_obj_mask = pred_obj_mask.cpu().numpy().astype(np.uint8)
        gt = compose_camera_gt_with_background(view, background)
        torchvision.utils.save_image(rendering, os.path.join(render_path, view.image_name + ".png"))
        torchvision.utils.save_image(gt, os.path.join(gts_path, view.image_name + ".png"))


def inpaint(dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool, skip_inpaint_render : bool, opt : OptimizationParams, select_obj_id : int, removal_thresh : float,  finetune_iteration: int, render_video : bool, args, config):
    """
    
    
    """
    scene_json_path = os.path.join(args.config_file)
    with open(scene_json_path, "r") as f:
        mask_info = json.load(f)
    args.circle_radius = mask_info.get("circle_radius")
    print("circle_radius: ", args.circle_radius)

    # 1. load gaussian checkpoint
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)
    args.supp_ply = str(
        resolve_support_ply(
            dataset.model_path,
            scene.loaded_iter,
            support_view_name=getattr(args, "support_view_name", None),
            explicit_support_ply=getattr(args, "supp_ply", None),
        )
    )
    print(f"Support ply: {args.supp_ply}")
    dataset.num_classes = args.num_classes
    print("Num classes: ", dataset.num_classes)
    
    classifier = torch.nn.Conv2d(gaussians.num_objects, dataset.num_classes, kernel_size=1)
    classifier.cuda()
    classifier.load_state_dict(torch.load(os.path.join(dataset.model_path,"point_cloud","iteration_"+str(scene.loaded_iter),"classifier.pth")))
    bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    views = scene.getTrainCameras()
    view = views[0]
    is_circle=True

    poses = generate_ellipse_path(views, n_frames=30, is_circle=is_circle, circle_radius=args.circle_radius) 
    virtual_pose_list = []
    unseen_mask_ready_dir = get_unseen_mask_ready_dir(dataset.model_path, args.target_id)
    ready_for_3dinpaint_color_dir = get_ready_for_3dinpaint_color_dir(dataset.model_path, args.target_id)
    for idx, pose in enumerate(tqdm(poses, desc="\nReplace real virtual camera views")):
        view_tmp = copy.deepcopy(view)
        view_tmp.world_view_transform = torch.tensor(getWorld2View2(pose[:3, :3].T, pose[:3, 3], view.trans, view.scale)).transpose(0, 1).cuda()
        view_tmp.projection_matrix = getProjectionMatrix(znear=view.znear, zfar=view.zfar, fovX=view.FoVx, fovY=view.FoVy).transpose(0,1).cuda()
        view_tmp.full_proj_transform = (view_tmp.world_view_transform.unsqueeze(0).bmm(view.projection_matrix.unsqueeze(0))).squeeze(0)
        view_tmp.camera_center = view_tmp.world_view_transform.inverse()[3, :3]
        
        view_tmp.image_name = f"{idx:05d}"

        object_path = unseen_mask_ready_dir / f"{view_tmp.image_name}.png"
        objects = Image.open(object_path) 
        view_tmp.objects = torch.from_numpy(np.array(objects)).to(view.data_device)

        image_path = find_image_for_stem(ready_for_3dinpaint_color_dir, view_tmp.image_name)
        image = Image.open(image_path)
        resolution=(view.image_width, view.image_height)
        resized_image_rgb = PILtoTorch(image, resolution)
        if resized_image_rgb.shape[0] == 1:
            resized_image_rgb = resized_image_rgb.repeat(3, 1, 1)
        gt_image = resized_image_rgb[:3, ...].clamp(0.0, 1.0).to(view.data_device)
        view_tmp.original_image = gt_image * torch.ones((1, view.image_height, view.image_width), device=view.data_device)
        if resized_image_rgb.shape[0] == 4:
            view_tmp.gt_alpha_mask = resized_image_rgb[3:4, ...].to(view.data_device)
            view_tmp.alpha_mask = view_tmp.gt_alpha_mask
        else:
            view_tmp.gt_alpha_mask = None
            view_tmp.alpha_mask = None

        view_tmp.R = pose[:3, :3].T
        view_tmp.T = pose[:3, 3]
        virtual_pose_list.append(view_tmp)

    # 2. inpaint selected object
    gaussians = finetune_inpaint(args, opt, dataset.model_path, scene.loaded_iter, virtual_pose_list, gaussians, pipeline, background, classifier, select_obj_id, scene.cameras_extent, removal_thresh, finetune_iteration)
   
    # 3. render new result
    scene = Scene(dataset, gaussians, load_iteration=f'_object_inpaint_virtual/iteration_'+str(finetune_iteration), shuffle=False)
    
    if render_video:
        render_video_func_wriva(dataset.source_path, dataset.model_path, scene.loaded_iter, scene.getTrainCameras(),
                                gaussians, pipeline, background, classifier, fps = 30)

    with torch.no_grad():
        if not skip_train:
            render_set(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background, classifier, args)

        if not skip_test:
            render_set(dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline, background, classifier, args)

        if "inpaint360" in args.source_path and not skip_inpaint_render:
            render_set(dataset.model_path, "inpaint", scene.loaded_iter, scene.getInpaintCameras(), gaussians, pipeline, background, classifier, args)

# Main Procedure
if __name__ == "__main__":
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    opt = OptimizationParams(parser)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--skip_inpaint_render", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--render_video", action="store_true")
    parser.add_argument('--temp_ply', type=str, default='temp.ply', help='The path to save the Temporary Gaussians.')
    parser.add_argument('--nb_points', type=int, default=100, help='Number of points for the remove_radius_outlier function.')
    parser.add_argument('--threshold', type=float, default=1.0, help='Threshold for the similar_points_tree function.')
    parser.add_argument('--radius', type=float, default=0.1, help='Radius for the remove_radius_outlier function.')
    parser.add_argument("--config_file", type=str, default="config/object_inpaint/inpaint360/doppelherz.json", help="Path to the configuration file")
    parser.add_argument("--supp_ply", type=str, default=None, help="Optional explicit support ply for initializing new inpaint gaussians.")
    parser.add_argument("--support_view_name", type=str, default=None, help="Optional virtual-view stem used to select support ply, e.g. 00014.")
    parser.add_argument("--storage_mode", choices=["full", "lite", "minimal"], default="full", help="Output retention mode.")
    args = get_combined_args(parser)

    # Read and parse the configuration file
    with open(args.config_file, 'r') as file:
        config = json.load(file)
    args.removal_thresh = config.get("removal_thresh")
    args.select_obj_id = config.get("select_obj_id")
    args.target_id = config.get("target_id")
    args.surrounding_ids = config.get("surrounding_ids")

    args.lambda_dssim = config.get("lambda_dssim")                
    args.finetune_iteration = config.get("finetune_iteration")
    args.opacity_init = config.get("opacity_init", 0.1)
    args.lambda_lpips = config.get("lambda_lpips")         

    scene_json_path = os.path.join(args.source_path, args.object_path, "scene.json")
    if not os.path.exists(scene_json_path):
        raise FileNotFoundError(
            f"scene.json not found at {scene_json_path}. "
            f"Check --source_path and --object_path."
        )
    with open(scene_json_path, "r") as f:
        scene_info = json.load(f)
    args.num_classes = scene_info.get("num_classes")

    torch.cuda.empty_cache()

    # Initialize system state (RNG)
    safe_state(args.quiet)

    inpaint(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test, args.skip_inpaint_render, opt.extract(args), args.select_obj_id, args.removal_thresh, args.finetune_iteration, args.render_video, args, config)
