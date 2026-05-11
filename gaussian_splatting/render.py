#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
from scene import Scene
import copy
import os
from tqdm import tqdm
import cv2
import numpy as np
from os import makedirs
from gaussian_renderer import render
import torchvision
try:
    import mediapy as media
except ImportError:
    media = None
from utils.general_utils import safe_state, compose_camera_gt_with_background
from utils.pose_utils import generate_ellipse_path
from utils.graphics_utils import getWorld2View2
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
try:
    from diff_gaussian_rasterization import SparseGaussianAdam
    SPARSE_ADAM_AVAILABLE = True
except:
    SPARSE_ADAM_AVAILABLE = False

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


def render_video_func_wriva(source_path, model_path, iteration, views, gaussians, pipeline, background,
                            fps=30, n_frames=240, output_root="video", legacy_video=True):

    print(f"\n Now we start to generate video!")
    render_path = os.path.join(model_path, output_root, "ours_{}".format(iteration))
    print(f"\n The video will be saved in {render_path}")
    makedirs(render_path, exist_ok=True)
    renders_path = os.path.join(render_path, "renders")
    makedirs(renders_path, exist_ok=True)

    render_poses = generate_ellipse_path(views, n_frames=n_frames)

    video_images_list = []
    for idx, pose in enumerate(tqdm(render_poses, desc="Rendering progress")):
        view = copy.deepcopy(views[0])
        view.image_name = "{0:05d}".format(idx)
        _set_camera_pose(view, pose)
        rendering = render(view, gaussians, pipeline, background)

        img = torch.clamp(rendering["render"], min=0., max=1.).cpu()
        torchvision.utils.save_image(img, os.path.join(renders_path, view.image_name + ".png"))
        if legacy_video:
            torchvision.utils.save_image(img, os.path.join(render_path, view.image_name + ".png"))

        video_img = _tensor_to_rgb8(img)
        video_images_list.append(video_img)

    if legacy_video:
        _write_video(os.path.join(render_path, "final_video.mp4"), video_images_list, fps)
    else:
        _write_video(os.path.join(render_path, "render_traj_color.mp4"), video_images_list, fps)


def render_set(model_path, name, iteration, views, gaussians, pipeline, background, train_test_exp, separate_sh):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")

    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)

    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        rendering = render(view, gaussians, pipeline, background, use_trained_exp=train_test_exp, separate_sh=separate_sh)["render"]
        gt = compose_camera_gt_with_background(view, background)

        if args.train_test_exp:
            rendering = rendering[..., rendering.shape[-1] // 2:]
            gt = gt[..., gt.shape[-1] // 2:]

        torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))

def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool,
                separate_sh: bool, render_video : bool, render_path : bool, render_path_frames : int,
                render_path_fps : int):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)

        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        if render_path:
            render_video_func_wriva(dataset.source_path, dataset.model_path, scene.loaded_iter, scene.getTrainCameras(),
                                    gaussians, pipeline, background, fps=render_path_fps,
                                    n_frames=render_path_frames, output_root="traj", legacy_video=False)

        if render_video:
            render_video_func_wriva(dataset.source_path, dataset.model_path, scene.loaded_iter, scene.getTrainCameras(),
                                    gaussians, pipeline, background, fps=render_path_fps,
                                    n_frames=render_path_frames, output_root="video", legacy_video=True)

        if not skip_train:
             render_set(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background, dataset.train_test_exp, separate_sh)

        if not skip_test:
             render_set(dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline, background, dataset.train_test_exp, separate_sh)

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
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test,
                SPARSE_ADAM_AVAILABLE, args.render_video, args.render_path, args.render_path_frames,
                args.render_path_fps)
