import numpy as np
from plyfile import PlyData, PlyElement
import torch
from sklearn.neighbors import KDTree

from simple_knn._C import distCUDA2
from torch import nn
import os


C0 = 0.28209479177387814
max_sh_degree = 3              

def mask_to_bbox(mask):
    # Find the rows and columns where the mask is non-zero
    rows = torch.any(mask, dim=1)
    cols = torch.any(mask, dim=0)
    ymin, ymax = torch.where(rows)[0][[0, -1]]    # height
    xmin, xmax = torch.where(cols)[0][[0, -1]]    # weight
    
    return xmin, ymin, xmax, ymax

def crop_using_bbox(image, bbox):
    xmin, ymin, xmax, ymax = bbox
    return image[:, ymin:ymax+1, xmin:xmax+1]

def RGB2SH(rgb):
    return (rgb - 0.5) / C0

def inverse_sigmoid(x):
    return torch.log(x/(1-x))

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

def create_from_pcd_our(pcd, path, args):
    
    fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()
    fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda())
    features = torch.zeros((fused_color.shape[0], 3, (max_sh_degree + 1) ** 2)).float().cuda()
    features[:, :3, 0 ] = fused_color
    features[:, 3:, 1:] = 0.0

    opacity_init = args.opacity_init

    # random init obj_id now
    fused_objects = RGB2SH(torch.rand((fused_point_cloud.shape[0], 16), device="cuda"))
    fused_objects = fused_objects[:,:,None] 

    print("Number of points at initialisation : ", fused_point_cloud.shape[0])

    dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()), 0.0000001)
    scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 3)
    rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
    rots[:, 0] = 1

    opacities = inverse_sigmoid(opacity_init * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))

    xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
    features_dc = nn.Parameter(features[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(True))
    features_rest = nn.Parameter(features[:,:,1:].transpose(1, 2).contiguous().requires_grad_(True))   #  
    scaling = nn.Parameter(scales.requires_grad_(True))
    rotation = nn.Parameter(rots.requires_grad_(True))
    opacity = nn.Parameter(opacities.requires_grad_(True))
    objects_dc = nn.Parameter(fused_objects.transpose(1, 2).contiguous().requires_grad_(False))
    max_radii2D = torch.zeros((xyz.shape[0]), device="cuda")

    xyz = xyz.detach().cpu().numpy()
    normals = np.zeros_like(xyz)
    f_dc = features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
    f_rest = features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
    opacities = opacity.detach().cpu().numpy()
    scale = scaling.detach().cpu().numpy()
    rotation = rotation.detach().cpu().numpy()
    obj_dc = objects_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()   # Gaussian Grouping

    dtype_full = [(attribute, 'f4') for attribute in construct_list_of_attributes(features_dc, features_rest, scaling, rotation, objects_dc)]

    elements = np.empty(xyz.shape[0], dtype=dtype_full)
    attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation, obj_dc), axis=1)
    elements[:] = list(map(tuple, attributes))
    el = PlyElement.describe(elements, 'vertex')
    PlyData([el]).write(path)

def load_ply_our(path, sub_features=None):
    """
    Load a PLY file and optionally initialize its features using KNN-based interpolation 
    from a reference set of sub-features.

    Args:
        path (str): Path to the .ply file containing Gaussian data.
        sub_features (dict, optional): Reference features used for spatial interpolation (KNN).
                                       Expected to contain 'xyz', 'scaling', 'rotation', etc.
    Returns:
        tuple: (xyz, features_dc, features_rest, opacity, scales, rots, objects_dc)
    """
    plydata = PlyData.read(path)
    
    xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                    np.asarray(plydata.elements[0]["y"]),
                    np.asarray(plydata.elements[0]["z"])),  axis=1)
    opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

    features_dc = np.zeros((xyz.shape[0], 3, 1))
    features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
    features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
    features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

    smarter_ini = True

    if smarter_ini:
        new_features = {}
        kdtree = KDTree(sub_features["xyz"].cpu().numpy())
        distances, indices = kdtree.query(xyz, k=5)
        # Initialize new points for each feature
        for key, feature in sub_features.items():
            # key  'xyz' 'features_dc' 'scaling' 'objects_dc' 'features_rest' 'opacity' 'rotation'
            feature_np = feature.cpu().numpy()
            
            # If we have valid neighbors, calculate the mean of neighbor points
            if feature_np.ndim == 2:
                neighbor_points = feature_np[indices]
            elif feature_np.ndim == 3:
                neighbor_points = feature_np[indices, :, :]
            else:
                raise ValueError(f"Unsupported feature dimension: {feature_np.ndim}")
            new_points_np = np.mean(neighbor_points, axis=1)   # knn feature
            
            # Convert back to tensor
            new_features[key] = new_points_np.astype(feature.cpu().numpy().dtype)

        new_features['xyz'] = xyz
        new_features['opacity'] = opacities
        new_features['features_dc'] = features_dc


        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))
        assert len(extra_f_names)==3*(max_sh_degree + 1) ** 2 - 3
        features_rest = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_rest[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        features_rest = features_rest.reshape((features_rest.shape[0], 3, (max_sh_degree + 1) ** 2 - 1))

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        os.remove(path)
        
        return new_features['xyz'], new_features['features_dc'], features_rest, new_features['opacity'], scales, rots, new_features['objects_dc'].transpose(0, 2, 1)
                                                                                             
    else:
        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))
        assert len(extra_f_names)==3*(max_sh_degree + 1) ** 2 - 3
        features_rest = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_rest[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        features_rest = features_rest.reshape((features_rest.shape[0], 3, (max_sh_degree + 1) ** 2 - 1))

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        # TODO TODO TODO 
        objects_dc = np.zeros((xyz.shape[0], 16, 1))          
        for idx in range(16):
            objects_dc[:, idx, 0] = np.asarray(plydata.elements[0]["obj_dc_"+str(idx)])

        os.remove(path)
        # xyz features_dc features_rest opacities scales rots
        return xyz, features_dc, features_rest, opacities, scales, rots, objects_dc


def similar_points_tree(point_cloud_A, point_cloud_B, threshold=1.0):

    tree = KDTree(point_cloud_B)
    distances, indices = tree.query(point_cloud_A, k=1)
    similar_indices = np.where(distances < threshold)[0]

    return similar_indices
