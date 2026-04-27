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

import os
import sys

# Add project root to PYTHONPATH
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

import torch
from scene import Scene
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
import numpy as np
from sklearn.cluster import KMeans, DBSCAN
import torch
import numpy as np

def filter_artifacts_by_kmeans(gaussians_object, n_clusters=2):

    xyz_tensor = gaussians_object._xyz
    xyz = xyz_tensor.detach().cpu().numpy()

    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(xyz)
    labels = kmeans.labels_

    unique_labels, counts = np.unique(labels, return_counts=True)
    main_cluster = unique_labels[np.argmax(counts)]

    keep_mask = (labels == main_cluster)
    filtered_xyz = xyz_tensor[keep_mask]
    keep_indices = torch.nonzero(torch.from_numpy(keep_mask)).squeeze(1)

    gaussians_object._xyz = gaussians_object._xyz[keep_indices]
    gaussians_object._features_dc = gaussians_object._features_dc[keep_indices]
    gaussians_object._features_rest = gaussians_object._features_rest[keep_indices]
    gaussians_object._opacity = gaussians_object._opacity[keep_indices]
    gaussians_object._scaling = gaussians_object._scaling[keep_indices]
    gaussians_object._rotation = gaussians_object._rotation[keep_indices]
    gaussians_object._objects_dc = gaussians_object._objects_dc[keep_indices]

    return gaussians_object


def filter_artifacts_by_dbscan(gaussians_object, eps=0.1, min_samples=10):
    xyz_tensor = gaussians_object._xyz
    xyz = xyz_tensor.detach().cpu().numpy()

    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(xyz)
    labels = clustering.labels_

    # label = -1 is noise
    if np.all(labels == -1):
        print("⚠️ Warning: all points are labeled as noise!")
        return gaussians_object  #

    valid_mask = labels != -1
    cluster_labels = labels[valid_mask]
    unique_labels, counts = np.unique(cluster_labels, return_counts=True)
    main_cluster = unique_labels[np.argmax(counts)]

    keep_mask = (labels == main_cluster)
    keep_indices = torch.nonzero(torch.from_numpy(keep_mask)).squeeze(1)

    gaussians_object._xyz = gaussians_object._xyz[keep_indices]
    gaussians_object._features_dc = gaussians_object._features_dc[keep_indices]
    gaussians_object._features_rest = gaussians_object._features_rest[keep_indices]
    gaussians_object._opacity = gaussians_object._opacity[keep_indices]
    gaussians_object._scaling = gaussians_object._scaling[keep_indices]
    gaussians_object._rotation = gaussians_object._rotation[keep_indices]
    gaussians_object._objects_dc = gaussians_object._objects_dc[keep_indices]

    return gaussians_object


def combine_gaussian(
    dataset: ModelParams,
    iteration: int,
    pipeline: PipelineParams,
    object_list,
    removal_iteration=None,
):
    """ 
    Combine the base scene with multiple removed objects into a single Gaussian model and save the result.

    Args:
        dataset (ModelParams): Model parameters containing SH degree and paths.
        iteration (int): The iteration number to load the base scene.
        pipeline (PipelineParams): Pipeline configuration for rendering.
        object_list (list): A list of object IDs to be re-integrated into the scene.

    Returns:
        GaussianModel: The final combined Gaussian model containing both the base scene and objects.
    """
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)

    if removal_iteration is None:
        removal_iteration = 2000

    for object_id in object_list:
        gaussians_object = GaussianModel(dataset.sh_degree)

        obj_model_path = f'_object_removal/iteration_{removal_iteration}/point_cloud_{object_id}.ply'
        scene_object = Scene(dataset, gaussians_object, load_iteration=obj_model_path, shuffle=False)

        gaussians_object=filter_artifacts_by_dbscan(gaussians_object)

        gaussians._xyz = torch.cat([gaussians._xyz, gaussians_object._xyz], dim=0)
        gaussians._features_dc = torch.cat([gaussians._features_dc, gaussians_object._features_dc], dim=0)
        gaussians._features_rest = torch.cat([gaussians._features_rest, gaussians_object._features_rest], dim=0)
        gaussians._opacity = torch.cat([gaussians._opacity, gaussians_object._opacity], dim=0)
        gaussians._scaling = torch.cat([gaussians._scaling, gaussians_object._scaling], dim=0)
        gaussians._rotation = torch.cat([gaussians._rotation, gaussians_object._rotation], dim=0)
        gaussians._objects_dc = torch.cat([gaussians._objects_dc, gaussians_object._objects_dc], dim=0)
    
    combined_gaussian_folder = os.path.dirname(scene.model_path + "/point_cloud" + iteration)
    combined_gaussian_path = os.path.join(combined_gaussian_folder, f"point_cloud.ply")
    gaussians.save_ply(combined_gaussian_path)
    print(f"The combined gaussian scene is saved at {combined_gaussian_path} ")
    return gaussians

 
if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default='_object_inpaint_virtual/iteration_4999/point_cloud.ply')
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--render_video", action="store_true")
    parser.add_argument("--object_list", help="e.g. [11,22]")
    parser.add_argument("--removal_iteration", default=2000, type=int, help="Removal-stage iteration used to load per-object ply files.")
    args = get_combined_args(parser)

    print("Rendering " + args.model_path)
    # Initialize system state (RNG)s
    safe_state(args.quiet)

    combine_gaussian(
        model.extract(args),
        args.iteration,
        pipeline.extract(args),
        args.object_list,
        removal_iteration=args.removal_iteration,
    )

    # python tools/combine_gaussian_scene.py -s data/inpaint360/fruits -m output/inpaint360/fruits
 
