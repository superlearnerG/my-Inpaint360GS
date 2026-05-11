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

from scene.cameras import Camera
import numpy as np
from utils.graphics_utils import fov2focal
from PIL import Image
import cv2

WARNED = False

def _open_image_preserve_alpha(image_path):
    image = Image.open(image_path)
    if image.mode in ("RGBA", "LA") or "transparency" in image.info:
        return image.convert("RGBA")
    if image.mode == "RGB":
        return image
    return image.convert("RGB")

def _raw_depth_array_2d(depth, depth_path):
    depth = np.asarray(depth)
    if depth.ndim == 3:
        if depth.shape[-1] == 1:
            depth = depth[..., 0]
        elif depth.shape[0] == 1:
            depth = depth[0]
        else:
            raise ValueError(f"Expected a single-channel raw depth map at '{depth_path}', got shape {depth.shape}.")
    if depth.ndim != 2:
        raise ValueError(f"Expected a 2D raw depth map at '{depth_path}', got shape {depth.shape}.")
    return depth.astype(np.float32, copy=False)

def _load_raw_invdepthmap(depth_path, depth_scale):
    raw_depth = _raw_depth_array_2d(np.load(depth_path), depth_path)
    scaled_depth = raw_depth * float(depth_scale)
    valid = np.isfinite(scaled_depth) & (scaled_depth > 0.0)
    invdepthmap = np.zeros_like(scaled_depth, dtype=np.float32)
    invdepthmap[valid] = 1.0 / np.maximum(scaled_depth[valid], 1e-6)
    return invdepthmap, valid.astype(np.float32)

def loadCam(args, id, cam_info, resolution_scale, is_nerf_synthetic, is_test_dataset):
    image = _open_image_preserve_alpha(cam_info.image_path)

    invdepthmask = None
    raw_depth_path = getattr(cam_info, "raw_depth_path", "")
    if raw_depth_path:
        try:
            invdepthmap, invdepthmask = _load_raw_invdepthmap(raw_depth_path, getattr(cam_info, "raw_depth_scale", 1.0))
        except FileNotFoundError:
            print(f"Error: The raw depth file at path '{raw_depth_path}' was not found.")
            raise
        except Exception as e:
            print(f"An unexpected error occurred when trying to read raw depth at {raw_depth_path}: {e}")
            raise
    elif cam_info.depth_path != "":
        try:
            if is_nerf_synthetic:
                invdepthmap = cv2.imread(cam_info.depth_path, -1).astype(np.float32) / 512
            else:
                invdepthmap = cv2.imread(cam_info.depth_path, -1).astype(np.float32) / float(2**16)

        except FileNotFoundError:
            print(f"Error: The depth file at path '{cam_info.depth_path}' was not found.")
            raise
        except IOError:
            print(f"Error: Unable to open the image file '{cam_info.depth_path}'. It may be corrupted or an unsupported format.")
            raise
        except Exception as e:
            print(f"An unexpected error occurred when trying to read depth at {cam_info.depth_path}: {e}")
            raise
    else:
        invdepthmap = None
        
    orig_w, orig_h = image.size
    if args.resolution in [1, 2, 4, 8]:
        resolution = round(orig_w/(resolution_scale * args.resolution)), round(orig_h/(resolution_scale * args.resolution))
    else:  # should be a type that converts to float
        if args.resolution == -1:
            if orig_w > 1600:
                global WARNED
                if not WARNED:
                    print("[ INFO ] Encountered quite large input images (>1.6K pixels width), rescaling to 1.6K.\n "
                        "If this is not desired, please explicitly specify '--resolution/-r' as 1")
                    WARNED = True
                global_down = orig_w / 1600
            else:
                global_down = 1
        else:
            global_down = orig_w / args.resolution
    

        scale = float(global_down) * float(resolution_scale)
        resolution = (int(orig_w / scale), int(orig_h / scale))

    return Camera(resolution, colmap_id=cam_info.uid, R=cam_info.R, T=cam_info.T, 
                  FoVx=cam_info.FovX, FoVy=cam_info.FovY, depth_params=cam_info.depth_params,
                  image=image, invdepthmap=invdepthmap,
                  image_name=cam_info.image_name, uid=id, data_device=args.data_device,
                  train_test_exp=args.train_test_exp, is_test_dataset=is_test_dataset, is_test_view=cam_info.is_test,
                  invdepthmask=invdepthmask)

def cameraList_from_camInfos(cam_infos, resolution_scale, args, is_nerf_synthetic, is_test_dataset):
    camera_list = []

    for id, c in enumerate(cam_infos):
        camera_list.append(loadCam(args, id, c, resolution_scale, is_nerf_synthetic, is_test_dataset))

    return camera_list

def camera_to_JSON(id, camera : Camera):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = camera.R.transpose()
    Rt[:3, 3] = camera.T
    Rt[3, 3] = 1.0

    W2C = np.linalg.inv(Rt)
    pos = W2C[:3, 3]
    rot = W2C[:3, :3]
    serializable_array_2d = [x.tolist() for x in rot]
    camera_entry = {
        'id' : id,
        'img_name' : camera.image_name,
        'width' : camera.width,
        'height' : camera.height,
        'position': pos.tolist(),
        'rotation': serializable_array_2d,
        'fy' : fov2focal(camera.FovY, camera.height),
        'fx' : fov2focal(camera.FovX, camera.width)
    }
    return camera_entry
