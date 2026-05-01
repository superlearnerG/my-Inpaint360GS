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

import os
import sys
from PIL import Image
from typing import NamedTuple
from scene.colmap_loader import read_extrinsics_text, read_intrinsics_text, qvec2rotmat, \
    read_extrinsics_binary, read_intrinsics_binary, read_points3D_binary, read_points3D_text, read_next_bytes
from utils.graphics_utils import getWorld2View2, focal2fov, fov2focal
import numpy as np
import json
from pathlib import Path
from plyfile import PlyData, PlyElement
from utils.sh_utils import SH2RGB
from scene.gaussian_model import BasicPointCloud

class CameraInfo(NamedTuple):
    uid: int
    R: np.array
    T: np.array
    FovY: np.array
    FovX: np.array
    depth_params: dict
    image_path: str
    image_name: str
    depth_path: str
    width: int
    height: int
    is_test: bool
    raw_depth_path: str = ""
    raw_depth_scale: float = 1.0

class SceneInfo(NamedTuple):
    point_cloud: BasicPointCloud
    train_cameras: list
    test_cameras: list
    nerf_normalization: dict
    ply_path: str
    is_nerf_synthetic: bool

def getNerfppNorm(cam_info):
    def get_center_and_diag(cam_centers):
        cam_centers = np.hstack(cam_centers)
        avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)
        center = avg_cam_center
        dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)
        diagonal = np.max(dist)
        return center.flatten(), diagonal

    cam_centers = []

    for cam in cam_info:
        W2C = getWorld2View2(cam.R, cam.T)
        C2W = np.linalg.inv(W2C)
        cam_centers.append(C2W[:3, 3:4])

    center, diagonal = get_center_and_diag(cam_centers)
    radius = diagonal * 1.1

    translate = -center

    return {"translate": translate, "radius": radius}

def _split_name_keys(text):
    stripped = str(text).strip()
    if not stripped:
        return set()
    basename = os.path.basename(stripped)
    stem = Path(basename).stem
    return {stripped, basename, stem}

def _read_split_list(list_path):
    split_names = set()
    with open(list_path, "r", encoding="utf-8") as file:
        for line in file:
            item = line.strip()
            if not item or item.startswith("#"):
                continue
            split_names.update(_split_name_keys(item))
    return split_names

def _name_in_split(image_name, split_names):
    return bool(_split_name_keys(image_name) & split_names)

def _load_dataset_split(path):
    train_list_path = os.path.join(path, "train_list.txt")
    test_list_path = os.path.join(path, "test_list.txt")
    has_train_list = os.path.exists(train_list_path)
    has_test_list = os.path.exists(test_list_path)
    if has_train_list and has_test_list:
        train_split_names = _read_split_list(train_list_path)
        test_split_names = _read_split_list(test_list_path)
        overlap = train_split_names & test_split_names
        if overlap:
            preview = ", ".join(sorted(overlap)[:10])
            raise ValueError(f"train_list.txt and test_list.txt overlap: {preview}")
        print(f"COLMAP split: using train_list.txt and test_list.txt from {path}")
        return {
            "mode": "list",
            "train_names": train_split_names,
            "test_names": test_split_names,
        }
    if has_train_list != has_test_list:
        print("Only one of train_list.txt/test_list.txt was found; falling back to basename holdout split.")
    return {"mode": "holdout", "train_names": set(), "test_names": set()}

def _basename_sequence_number(image_name):
    stem = Path(os.path.basename(str(image_name))).stem
    try:
        return int(stem)
    except ValueError:
        pass
    digits = []
    for char in reversed(stem):
        if not char.isdigit():
            break
        digits.append(char)
    if not digits:
        return None
    return int("".join(reversed(digits)))

def _is_holdout_test_image(image_name, llffhold):
    sequence_number = _basename_sequence_number(image_name)
    return sequence_number is not None and sequence_number % llffhold == 0

def _depth_array_2d(depth, depth_path):
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
    return depth

def _resolve_raw_depth_folder(path, depths, use_depth_loss):
    if not use_depth_loss:
        return ""
    depth_dir = depths if depths else "depth"
    depth_dir = os.path.expanduser(str(depth_dir))
    if os.path.isabs(depth_dir):
        depth_folder = depth_dir
    elif os.path.isdir(depth_dir):
        depth_folder = os.path.abspath(depth_dir)
    else:
        depth_folder = os.path.join(path, depth_dir)
    if not os.path.isdir(depth_folder):
        raise FileNotFoundError(f"--use_depth_loss expects raw .npy depth maps under '{depth_folder}'.")
    print(f"[Depth Loss] Loading raw depth maps from {depth_folder}")
    return depth_folder

def _raw_depth_path(depths_folder, image_name):
    if depths_folder == "":
        return ""
    return os.path.join(depths_folder, f"{Path(image_name).stem}.npy")

def _read_points3d_xyz_by_id_binary(path):
    points = {}
    with open(path, "rb") as fid:
        num_points = read_next_bytes(fid, 8, "Q")[0]
        for _ in range(num_points):
            point_props = read_next_bytes(fid, 43, "QdddBBBd")
            point_id = int(point_props[0])
            points[point_id] = np.array(point_props[1:4], dtype=np.float64)
            track_length = read_next_bytes(fid, 8, "Q")[0]
            fid.seek(8 * track_length, 1)
    return points

def _read_points3d_xyz_by_id_text(path):
    points = {}
    with open(path, "r") as fid:
        for line in fid:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            elems = line.split()
            points[int(elems[0])] = np.array(tuple(map(float, elems[1:4])), dtype=np.float64)
    return points

def _read_points3d_xyz_by_id(path):
    bin_path = os.path.join(path, "sparse/0/points3D.bin")
    txt_path = os.path.join(path, "sparse/0/points3D.txt")
    if os.path.exists(bin_path):
        try:
            return _read_points3d_xyz_by_id_binary(bin_path)
        except Exception:
            if not os.path.exists(txt_path):
                raise
    if os.path.exists(txt_path):
        return _read_points3d_xyz_by_id_text(txt_path)
    raise FileNotFoundError(f"COLMAP points3D file not found under {os.path.join(path, 'sparse/0')}")

def _select_evenly_spaced(items, max_count):
    if len(items) <= max_count:
        return items
    indices = np.linspace(0, len(items) - 1, max_count, dtype=int)
    return [items[int(idx)] for idx in indices]

def _estimate_depth_scale_from_colmap(path, cam_extrinsics, depths_folder, max_views=32, max_points_per_view=12000):
    xyz_by_id = _read_points3d_xyz_by_id(path)
    ratios = []
    used_views = 0

    extrinsics = sorted(cam_extrinsics.values(), key=lambda extr: extr.name)
    for extr in _select_evenly_spaced(extrinsics, max_views):
        depth_path = _raw_depth_path(depths_folder, extr.name)
        if not os.path.exists(depth_path):
            continue

        point_ids = np.asarray(extr.point3D_ids)
        xys = np.asarray(extr.xys)
        valid_indices = np.flatnonzero(point_ids != -1)
        if valid_indices.size == 0:
            continue
        if valid_indices.size > max_points_per_view:
            valid_indices = np.asarray(_select_evenly_spaced(valid_indices.tolist(), max_points_per_view))

        matched_xys = []
        matched_xyz = []
        for idx in valid_indices:
            xyz = xyz_by_id.get(int(point_ids[idx]))
            if xyz is None:
                continue
            matched_xys.append(xys[idx])
            matched_xyz.append(xyz)
        if not matched_xyz:
            continue

        raw_depth = _depth_array_2d(np.load(depth_path, mmap_mode="r"), depth_path)
        matched_xys = np.asarray(matched_xys, dtype=np.float64)
        matched_xyz = np.asarray(matched_xyz, dtype=np.float64)
        u = np.rint(matched_xys[:, 0]).astype(np.int64)
        v = np.rint(matched_xys[:, 1]).astype(np.int64)
        in_image = (u >= 0) & (v >= 0) & (u < raw_depth.shape[1]) & (v < raw_depth.shape[0])
        if not np.any(in_image):
            continue

        R = qvec2rotmat(extr.qvec)
        t = np.asarray(extr.tvec, dtype=np.float64)
        z_colmap = (R @ matched_xyz[in_image].T).T[:, 2] + t[2]
        raw_z = np.asarray(raw_depth[v[in_image], u[in_image]], dtype=np.float64)
        valid = np.isfinite(raw_z) & (raw_z > 0.0) & np.isfinite(z_colmap) & (z_colmap > 0.0)
        view_ratios = z_colmap[valid] / raw_z[valid]
        view_ratios = view_ratios[np.isfinite(view_ratios) & (view_ratios > 0.0)]
        if view_ratios.size == 0:
            continue
        ratios.append(view_ratios)
        used_views += 1

    if not ratios:
        raise RuntimeError(
            "Unable to estimate --depth_scale from COLMAP tracks and raw depth maps. "
            "Pass a positive --depth_scale manually."
        )

    ratios = np.concatenate(ratios)
    if ratios.size < 100:
        raise RuntimeError(
            f"Only {ratios.size} valid COLMAP/raw-depth correspondences were found; "
            "pass a positive --depth_scale manually."
        )

    scale = float(np.median(ratios))
    print(
        "[Depth Loss] Estimated raw-depth scale from COLMAP tracks: "
        f"{scale:.6f} ({ratios.size} samples from {used_views} views; "
        f"p05={np.percentile(ratios, 5):.6f}, p95={np.percentile(ratios, 95):.6f})"
    )
    return scale

def _resolve_depth_scale(path, cam_extrinsics, depths_folder, requested_depth_scale, use_depth_loss):
    if not use_depth_loss:
        return 1.0
    requested_depth_scale = float(requested_depth_scale)
    if requested_depth_scale > 0.0:
        print(f"[Depth Loss] Using manual raw-depth scale: {requested_depth_scale:.6f}")
        return requested_depth_scale
    return _estimate_depth_scale_from_colmap(path, cam_extrinsics, depths_folder)

def _validate_train_raw_depth_paths(train_cam_infos):
    missing = [
        cam.image_name
        for cam in train_cam_infos
        if not getattr(cam, "raw_depth_path", "") or not os.path.exists(getattr(cam, "raw_depth_path", ""))
    ]
    if missing:
        preview = ", ".join(missing[:10])
        suffix = "" if len(missing) <= 10 else f", ... ({len(missing)} missing)"
        raise FileNotFoundError(f"Missing raw .npy depth maps for training views: {preview}{suffix}")

def readColmapCameras(cam_extrinsics, cam_intrinsics, depths_params, images_folder, depths_folder, test_cam_names_list,
                      raw_depths_folder="", raw_depth_scale=1.0):
    cam_infos = []
    for idx, key in enumerate(cam_extrinsics):
        sys.stdout.write('\r')
        # the exact output you're looking for:
        sys.stdout.write("Reading camera {}/{}".format(idx+1, len(cam_extrinsics)))
        sys.stdout.flush()

        extr = cam_extrinsics[key]
        intr = cam_intrinsics[extr.camera_id]
        height = intr.height
        width = intr.width

        uid = intr.id
        R = np.transpose(qvec2rotmat(extr.qvec))
        T = np.array(extr.tvec)

        if intr.model=="SIMPLE_PINHOLE":
            focal_length_x = intr.params[0]
            FovY = focal2fov(focal_length_x, height)
            FovX = focal2fov(focal_length_x, width)
        elif intr.model=="PINHOLE":
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[1]
            FovY = focal2fov(focal_length_y, height)
            FovX = focal2fov(focal_length_x, width)
        else:
            assert False, "Colmap camera model not handled: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras) supported!"

        n_remove = len(extr.name.split('.')[-1]) + 1
        depth_params = None
        if depths_params is not None:
            try:
                depth_params = depths_params[extr.name[:-n_remove]]
            except:
                print("\n", key, "not found in depths_params")

        image_path = os.path.join(images_folder, extr.name)
        image_name = extr.name
        depth_path = os.path.join(depths_folder, f"{extr.name[:-n_remove]}.png") if depths_folder != "" else ""
        raw_depth_path = _raw_depth_path(raw_depths_folder, image_name) if raw_depths_folder != "" else ""

        cam_info = CameraInfo(uid=uid, R=R, T=T, FovY=FovY, FovX=FovX, depth_params=depth_params,
                              image_path=image_path, image_name=image_name, depth_path=depth_path,
                              width=width, height=height, is_test=image_name in test_cam_names_list,
                              raw_depth_path=raw_depth_path, raw_depth_scale=raw_depth_scale)
        cam_infos.append(cam_info)

    sys.stdout.write('\n')
    return cam_infos

def fetchPly(path):
    plydata = PlyData.read(path)
    vertices = plydata['vertex']
    positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
    colors = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T / 255.0
    normals = np.vstack([vertices['nx'], vertices['ny'], vertices['nz']]).T
    return BasicPointCloud(points=positions, colors=colors, normals=normals)

def storePly(path, xyz, rgb):
    # Define the dtype for the structured array
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
            ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
    
    normals = np.zeros_like(xyz)

    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals, rgb), axis=1)
    elements[:] = list(map(tuple, attributes))

    # Create the PlyData object and write to file
    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element])
    ply_data.write(path)

def readColmapSceneInfo(args, images, depths, eval, train_test_exp, llffhold=8):
    path = args.source_path
    try:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.bin")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.bin")
        cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
    except:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.txt")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.txt")
        cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_text(cameras_intrinsic_file)

    use_depth_loss = getattr(args, "use_depth_loss", False)
    raw_depths_folder = _resolve_raw_depth_folder(path, depths, use_depth_loss)
    raw_depth_scale = _resolve_depth_scale(
        path, cam_extrinsics, raw_depths_folder, getattr(args, "depth_scale", 0.0), use_depth_loss
    )

    depth_params_file = os.path.join(path, "sparse/0", "depth_params.json")
    ## if depth_params_file isnt there AND depths file is here -> throw error
    depths_params = None
    if depths != "" and not use_depth_loss:
        try:
            with open(depth_params_file, "r") as f:
                depths_params = json.load(f)
            all_scales = np.array([depths_params[key]["scale"] for key in depths_params])
            if (all_scales > 0).sum():
                med_scale = np.median(all_scales[all_scales > 0])
            else:
                med_scale = 0
            for key in depths_params:
                depths_params[key]["med_scale"] = med_scale

        except FileNotFoundError:
            print(f"Error: depth_params.json file not found at path '{depth_params_file}'.")
            sys.exit(1)
        except Exception as e:
            print(f"An unexpected error occurred when trying to open depth_params.json file: {e}")
            sys.exit(1)

    split_config = _load_dataset_split(path) if eval else {"mode": "disabled", "train_names": set(), "test_names": set()}
    if eval:
        if "360" in path:
            llffhold = 8
        cam_names = sorted([cam_extrinsics[cam_id].name for cam_id in cam_extrinsics])
        if split_config["mode"] == "list":
            test_cam_names_list = [name for name in cam_names if _name_in_split(name, split_config["test_names"])]
        elif llffhold:
            print("\n ------------LLFF HOLD-------------")
            print(f"COLMAP split: train_list.txt/test_list.txt not found; using basename index % {llffhold} == 0 as test")
            if "inpaint360" in args.source_path:
                print("\n We are using our inpaint360 dataset.")
                train_and_test_cam_names = [name for name in cam_names if "test" not in name]             # inpainting gt
                test_cam_names_list = [name for name in train_and_test_cam_names if _is_holdout_test_image(name, llffhold)]
            else:
                test_cam_names_list = [name for name in cam_names if _is_holdout_test_image(name, llffhold)]
            if not test_cam_names_list:
                print(f"Warning: no numeric basename matched index % {llffhold} == 0; test set is empty.")
        else:
            with open(os.path.join(path, "sparse/0", "test.txt"), 'r') as file:
                test_cam_names_list = [line.strip() for line in file]

    else:
        test_cam_names_list = []

    reading_dir = "images" if images == None else images
    legacy_depths_folder = os.path.join(path, depths) if depths != "" and not use_depth_loss else ""
    cam_infos_unsorted = readColmapCameras(
        cam_extrinsics=cam_extrinsics, cam_intrinsics=cam_intrinsics, depths_params=depths_params,
        images_folder=os.path.join(path, reading_dir), 
        depths_folder=legacy_depths_folder, test_cam_names_list=test_cam_names_list,
        raw_depths_folder=raw_depths_folder, raw_depth_scale=raw_depth_scale)  # test images
    cam_infos = sorted(cam_infos_unsorted.copy(), key = lambda x : x.image_name)

    if eval and split_config["mode"] == "list":
        train_cam_infos = [c for c in cam_infos if _name_in_split(c.image_name, split_config["train_names"])]
        test_cam_infos = [c for c in cam_infos if _name_in_split(c.image_name, split_config["test_names"])]
        if not train_cam_infos:
            raise ValueError(f"No COLMAP cameras matched {os.path.join(path, 'train_list.txt')}.")
        if not test_cam_infos:
            raise ValueError(f"No COLMAP cameras matched {os.path.join(path, 'test_list.txt')}.")
        inpaint_cam_infos = []
    elif "inpaint360" in args.source_path:
        train_inpaint_cam_infos = [c for c in cam_infos if train_test_exp or not c.is_test]
        train_cam_infos = [c for c in train_inpaint_cam_infos if "test" not in c.image_name]
        inpaint_cam_infos = [c for c in train_inpaint_cam_infos if "test" in c.image_name]   
        test_cam_infos = [c for c in cam_infos if c.is_test]
    else:
        train_cam_infos = [c for c in cam_infos if train_test_exp or not c.is_test]
        test_cam_infos = [c for c in cam_infos if c.is_test]

    if use_depth_loss:
        _validate_train_raw_depth_paths(train_cam_infos)

    print("\nTraining images:     ", len(train_cam_infos))
    print("Testing images:     ", len(test_cam_infos))
    if "inpaint360" in args.source_path:
        print("Inpainting images:     ", len(inpaint_cam_infos))
    
    print("\n Those are Training Images!")
    print([c.image_name for c in train_cam_infos])
    print("\n Those are Test Images!")
    print([c.image_name for c in test_cam_infos])
    if "inpaint360" in args.source_path:
        print("\n Those are Inpainting Images!")
        print([c.image_name for c in inpaint_cam_infos])

    nerf_normalization = getNerfppNorm(train_cam_infos)

    if args.init_mode == "dense":
        ply_path = os.path.join(path, "dense/fused.ply")
        print('\n We train the model with a dense point cloud init.')
    elif args.init_mode == "sparse":
        ply_path = os.path.join(path, "sparse/0/points3D.ply")
    bin_path = os.path.join(path, "sparse/0/points3D.bin")
    txt_path = os.path.join(path, "sparse/0/points3D.txt")
    if not os.path.exists(ply_path):
        print("Converting point3d.bin to .ply, will happen only the first time you open the scene.")
        try:
            xyz, rgb, _ = read_points3D_binary(bin_path)
        except:
            xyz, rgb, _ = read_points3D_text(txt_path)
        storePly(ply_path, xyz, rgb)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path,
                           is_nerf_synthetic=False)
    return scene_info

def readCamerasFromTransforms(path, transformsfile, depths_folder, white_background, is_test, extension=".png"):
    cam_infos = []

    with open(os.path.join(path, transformsfile)) as json_file:
        contents = json.load(json_file)
        fovx = contents["camera_angle_x"]

        frames = contents["frames"]
        for idx, frame in enumerate(frames):
            cam_name = os.path.join(path, frame["file_path"] + extension)

            # NeRF 'transform_matrix' is a camera-to-world transform
            c2w = np.array(frame["transform_matrix"])
            # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
            c2w[:3, 1:3] *= -1

            # get the world-to-camera transform and set R, T
            w2c = np.linalg.inv(c2w)
            R = np.transpose(w2c[:3,:3])  # R is stored transposed due to 'glm' in CUDA code
            T = w2c[:3, 3]

            image_path = os.path.join(path, cam_name)
            image_name = Path(cam_name).stem
            image = Image.open(image_path)

            im_data = np.array(image.convert("RGBA"))

            bg = np.array([1,1,1]) if white_background else np.array([0, 0, 0])

            norm_data = im_data / 255.0
            arr = norm_data[:,:,:3] * norm_data[:, :, 3:4] + bg * (1 - norm_data[:, :, 3:4])
            image = Image.fromarray(np.array(arr*255.0, dtype=np.byte), "RGB")

            fovy = focal2fov(fov2focal(fovx, image.size[0]), image.size[1])
            FovY = fovy 
            FovX = fovx

            depth_path = os.path.join(depths_folder, f"{image_name}.png") if depths_folder != "" else ""

            cam_infos.append(CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX,
                            image_path=image_path, image_name=image_name,
                            width=image.size[0], height=image.size[1], depth_path=depth_path, depth_params=None, is_test=is_test))
            
    return cam_infos

def readNerfSyntheticInfo(path, white_background, depths, eval, extension=".png"):

    depths_folder=os.path.join(path, depths) if depths != "" else ""
    print("Reading Training Transforms")
    train_cam_infos = readCamerasFromTransforms(path, "transforms_train.json", depths_folder, white_background, False, extension)
    print("Reading Test Transforms")
    test_cam_infos = readCamerasFromTransforms(path, "transforms_test.json", depths_folder, white_background, True, extension)
    
    if not eval:
        train_cam_infos.extend(test_cam_infos)
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "points3d.ply")
    if not os.path.exists(ply_path):
        # Since this data set has no colmap data, we start with random points
        num_pts = 100_000
        print(f"Generating random point cloud ({num_pts})...")
        
        # We create random points inside the bounds of the synthetic Blender scenes
        xyz = np.random.random((num_pts, 3)) * 2.6 - 1.3
        shs = np.random.random((num_pts, 3)) / 255.0
        pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3)))

        storePly(ply_path, xyz, SH2RGB(shs) * 255)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path,
                           is_nerf_synthetic=True)
    return scene_info

sceneLoadTypeCallbacks = {
    "Colmap": readColmapSceneInfo,
    "Blender" : readNerfSyntheticInfo
}
