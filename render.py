# Copyright (C) 2023, Gaussian-Grouping
# Gaussian-Grouping research group, https://github.com/lkeab/gaussian-grouping
# All rights reserved.
#
# ------------------------------------------------------------------------
# Modified from codes in Gaussian-Splatting 
# GRAPHDECO research group, https://team.inria.fr/graphdeco

import torch
from scene import Scene
import copy
import os
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
import torchvision
from utils.general_utils import safe_state, compose_camera_gt_with_background
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
from utils.point_utils import create_point_cloud, ply_color_fusion, get_intrinsics
from utils.pose_utils import generate_ellipse_path
from utils.graphics_utils import getWorld2View2
import numpy as np
from PIL import Image
import colorsys
import cv2
from sklearn.decomposition import PCA
import json
try:
    import mediapy as media
except ImportError:
    media = None


def id2rgb(id, max_num_obj=256): 
  
    if not 0 <= id <= max_num_obj:
        raise ValueError("ID should be in range(0, max_num_obj)")

    # Convert the ID into a hue value
    golden_ratio = 2.7182818284   
    h = ((id * golden_ratio) % 1)      
    s = 0.5 + (id % 2) * 0.5           
    l = 0.5

    # Use colorsys to convert HSL to RGB
    rgb = np.zeros((3, ), dtype=np.uint8)   
    if id==0:                              
        return rgb
    r, g, b = colorsys.hls_to_rgb(h, l, s)
    rgb[0], rgb[1], rgb[2] = int(r*255), int(g*255), int(b*255)

    return rgb

def visualize_obj(objects):
    
    rgb_mask = np.zeros((*objects.shape[-2:], 3), dtype=np.uint8)
    all_obj_ids = np.unique(objects)
    for id in all_obj_ids:
        colored_mask = id2rgb(id)             
        rgb_mask[objects == id] = colored_mask      
    return rgb_mask


def render_set(model_path, name, iteration, views, gaussians, pipeline, background, classifier, storage_mode="full"):
    
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")           
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")
    gt_obj_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt_objects_color")
    pred_obj_path = os.path.join(model_path, name, "ours_{}".format(iteration), "objects_pred")
    pred_obj_color_path = os.path.join(model_path, name, "ours_{}".format(iteration), "objects_pred_color")
    depth_path=os.path.join(model_path, name, "ours_{}".format(iteration), "depth")
    save_fused_ply = storage_mode == "full"
    fused_full_col_dep_ply_path=os.path.join(model_path, name, "ours_{}".format(iteration), "fused_full_col_dep_ply") 
   
    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)
    makedirs(gt_obj_path, exist_ok=True)
    makedirs(pred_obj_path, exist_ok=True)
    makedirs(pred_obj_color_path, exist_ok=True)
    makedirs(depth_path, exist_ok=True)
    if save_fused_ply:
        makedirs(fused_full_col_dep_ply_path, exist_ok=True)

        print(f"\nWe save our inpainted fused-full-depth-color ply at {fused_full_col_dep_ply_path}.\n")
    for i, view in enumerate(tqdm(views, desc="Rendering progress")):
        idx = i + 1  
        results = render(view, gaussians, pipeline, background)
        rendering = results["render"]              
        rendering_obj = results["render_object"]   
        
        logits = classifier(rendering_obj)         
        pred_obj_mask = torch.argmax(logits,dim=0)
        pred_obj_color_mask = visualize_obj(pred_obj_mask.cpu().numpy().astype(np.uint8))    
        
        gt_objects = view.objects
        gt_obj_mask = visualize_obj(gt_objects.cpu().numpy().astype(np.uint8)) 

        depth=results["depth_3dgs"].squeeze(0).detach().cpu().numpy()
       
        np.save(os.path.join(depth_path, view.image_name+".npy"),depth)
        depth = (depth - depth.min()) / (depth.max() - depth.min())   
        depth = (depth * 255.0).astype(np.uint8)
        depth = cv2.applyColorMap(depth, cv2.COLORMAP_JET)
        cv2.imwrite(os.path.join(depth_path, view.image_name + ".png"), depth)
        
        Image.fromarray(gt_obj_mask).save(os.path.join(gt_obj_path, view.image_name + ".png"))
        pred_obj_mask = pred_obj_mask.cpu().numpy().astype(np.uint8)  
        Image.fromarray(pred_obj_mask).save(os.path.join(pred_obj_path, view.image_name + ".png"))
        Image.fromarray(pred_obj_color_mask).save(os.path.join(pred_obj_color_path, view.image_name + ".png"))
        gt = compose_camera_gt_with_background(view, background)
        torchvision.utils.save_image(rendering, os.path.join(render_path, view.image_name + ".png"))
        torchvision.utils.save_image(gt, os.path.join(gts_path, view.image_name + ".png"))

        # save point cloud with color
        if save_fused_ply:
            w2c = np.zeros((4, 4))
            w2c[:3, :3] = view.R.transpose()    # view.R: camera to world
            w2c[:3, 3] = view.T                 # view.T: world to camera
            w2c[3, 3] = 1.0
            c2w = np.linalg.inv(w2c)
            intrinsics = get_intrinsics(view.image_height, view.image_width,view.FoVx,view.FoVy)
            points = create_point_cloud(results["depth_3dgs"].squeeze(0).detach().cpu().numpy(), intrinsics, c2w)
            ply_path = os.path.join(fused_full_col_dep_ply_path, view.image_name+".ply")
            colors = cv2.imread(os.path.join(gts_path, view.image_name+".png")).reshape(-1,3)
            ply_color_fusion(points, colors, ply_path, mask=None)



def _set_camera_pose(view, pose):
    view.R = pose[:3, :3].T
    view.T = pose[:3, 3]
    view.world_view_transform = torch.tensor(getWorld2View2(view.R, view.T, view.trans, view.scale)).transpose(0, 1).cuda()
    view.full_proj_transform = (view.world_view_transform.unsqueeze(0).bmm(view.projection_matrix.unsqueeze(0))).squeeze(0)
    view.camera_center = view.world_view_transform.inverse()[3, :3]


def _tensor_to_rgb8(image):
    return (torch.clamp(image, 0.0, 1.0).permute(1, 2, 0).detach().cpu().numpy() * 255.0).astype(np.uint8)


def _write_video(video_path, frames, fps):
    if not frames:
        return
    frames = [frame[:frame.shape[0] // 2 * 2, :frame.shape[1] // 2 * 2] for frame in frames]
    if media is not None:
        with media.VideoWriter(video_path, shape=frames[0].shape[:2], codec="h264", fps=fps, crf=18, input_format="rgb") as writer:
            for frame in frames:
                writer.add_image(frame)
        return

    height, width = frames[0].shape[:2]
    writer = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    for frame in frames:
        writer.write(frame[:, :, ::-1])
    writer.release()


def render_path_video(source_path, model_path, iteration, views, gaussians, pipeline, background, classifier,
                      n_frames=240, fps=30, output_root="traj", legacy_video=False):

    render_path = os.path.join(model_path, output_root, "ours_{}".format(iteration))
    print(f"\nThe trajectory video will be saved in {render_path}")
    renders_path = os.path.join(render_path, "renders")
    pred_obj_color_path = os.path.join(render_path, "objects_pred_color")
    combined_path = os.path.join(render_path, "combined")
    makedirs(renders_path, exist_ok=True)
    makedirs(pred_obj_color_path, exist_ok=True)
    makedirs(combined_path, exist_ok=True)

    render_poses = generate_ellipse_path(views, n_frames=n_frames)
    color_frames, object_frames, combined_frames = [], [], []
    for idx, pose in enumerate(tqdm(render_poses, desc="Rendering progress")):
        view = copy.deepcopy(views[0])
        view.image_name = "{0:05d}".format(idx)
        _set_camera_pose(view, pose)
        rendering = render(view, gaussians, pipeline, background)

        image = torch.clamp(rendering["render"], min=0., max=1.).cpu()
        rgb_frame = _tensor_to_rgb8(image)
        torchvision.utils.save_image(image, os.path.join(renders_path, view.image_name + ".png"))

        rendering_obj = rendering["render_object"]
        logits = classifier(rendering_obj)
        pred_obj = torch.argmax(logits, dim=0)
        pred_obj_frame = visualize_obj(pred_obj.cpu().numpy().astype(np.uint8))
        Image.fromarray(pred_obj_frame).save(os.path.join(pred_obj_color_path, view.image_name + ".png"))

        combined_frame = np.concatenate([rgb_frame, pred_obj_frame], axis=1)
        Image.fromarray(combined_frame).save(os.path.join(combined_path, view.image_name + ".png"))
        if legacy_video:
            Image.fromarray(combined_frame).save(os.path.join(render_path, view.image_name + ".png"))

        color_frames.append(rgb_frame)
        object_frames.append(pred_obj_frame)
        combined_frames.append(combined_frame)

    if legacy_video:
        _write_video(os.path.join(render_path, "final_video.mp4"), combined_frames, fps)
    else:
        _write_video(os.path.join(render_path, "render_traj_color.mp4"), color_frames, fps)
        _write_video(os.path.join(render_path, "render_traj_objects.mp4"), object_frames, fps)
        _write_video(os.path.join(render_path, "render_traj_combined.mp4"), combined_frames, fps)


def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool,
                render_video : bool, render_path : bool, render_path_frames : int, render_path_fps : int,
                storage_mode: str = "full"):

    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)
        
        scene_json_path = os.path.join(dataset.source_path, dataset.object_path, "scene.json")
        if not os.path.exists(scene_json_path):
            raise FileNotFoundError(
                f"scene.json not found at {scene_json_path}. "
                f"Check --source_path and --object_path."
            )
        with open(scene_json_path, "r") as f:
            mask_info = json.load(f)
        dataset.num_classes = mask_info.get("num_classes")
        print("Num classes: ", dataset.num_classes)

        classifier = torch.nn.Conv2d(gaussians.num_objects, dataset.num_classes, kernel_size=1)
        classifier.cuda()
        classifier.load_state_dict(torch.load(os.path.join(dataset.model_path,"point_cloud","iteration_"+str(scene.loaded_iter),"classifier.pth")))

        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        if render_path:
            render_path_video(dataset.source_path, dataset.model_path, scene.loaded_iter, scene.getTrainCameras(),
                              gaussians, pipeline, background, classifier,
                              n_frames=render_path_frames, fps=render_path_fps, output_root="traj")

        if render_video:
            render_path_video(dataset.source_path, dataset.model_path, scene.loaded_iter, scene.getTrainCameras(),
                              gaussians, pipeline, background, classifier,
                              n_frames=render_path_frames, fps=render_path_fps, output_root="video", legacy_video=True)

        if not skip_train:
             render_set(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background, classifier, storage_mode=storage_mode)

        if (not skip_test) and (len(scene.getTestCameras()) > 0):
             render_set(dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline, background, classifier, storage_mode=storage_mode)

        # if "inpaint360" in args.source_path:
        #     from edit_object_inpaint import render_set as render_set_inpaint
        #     print(scene.loaded_iter)
        #     render_set_inpaint(dataset.model_path, "inpaint", scene.loaded_iter, scene.getInpaintCameras(), gaussians, pipeline, background, classifier, args)

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--render_video", action="store_true")
    parser.add_argument("--render_path", action="store_true")
    parser.add_argument("--render_path_frames", default=240, type=int)
    parser.add_argument("--render_path_fps", default=30, type=int)
    parser.add_argument("--storage_mode", choices=["full", "lite", "minimal"], default="full", help="Output retention mode.")
    args = get_combined_args(parser)

    safe_state(args.quiet)

    render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test,
                args.render_video, args.render_path, args.render_path_frames, args.render_path_fps, args.storage_mode)
