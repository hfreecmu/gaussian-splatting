# import os
# import math
import torch
#import viser.transforms as vst
# from scipy.spatial.transform import Rotation

import einops
from e3nn import o3
from einops import einsum

# from scene import Scene
# from gaussian_renderer import GaussianModel
from utils.gaussian_utils import GaussianTransformUtils

# from argparse import ArgumentParser
# from arguments import ModelParams, PipelineParams, get_combined_args

import pytorch3d.transforms


# def searchForMaxIteration(folder):
#     saved_iters = [int(fname.split("_")[-1]) for fname in os.listdir(folder)]
    
#     return max(saved_iters)

# def transform_shs(shs_feat, rotation_matrix):
#     P = torch.tensor([[0, 0, 1], [1, 0, 0], [0, 1, 0]],dtype=torch.float32,device=rotation_matrix.device) 
#     permuted_rotation_matrix = torch.linalg.inv(P) @ rotation_matrix @ P
#     #TODO I harry maybe modified this
#     rotation_angles = o3._rotation.matrix_to_angles(permuted_rotation_matrix)
#     #rotation_angles = pytorch3d.transforms.matrix_to_euler_angles(permuted_rotation_matrix,'YXY')

#     #TODO I harry modified these function libraries to support cuda
#     D1 = o3.wigner_D(1, rotation_angles[0], - rotation_angles[1], rotation_angles[2])
#     D2 = o3.wigner_D(2, rotation_angles[0], - rotation_angles[1], rotation_angles[2])
#     D3 = o3.wigner_D(3, rotation_angles[0], - rotation_angles[1], rotation_angles[2])

#     one_degree_shs = shs_feat[:, 0:3]
#     one_degree_shs = einops.rearrange(one_degree_shs, 'n shs_num rgb -> n rgb shs_num')
#     one_degree_shs = einsum(
#             D1,
#             one_degree_shs,
#             "... i j, ... j -> ... i",
#         )
#     one_degree_shs = einops.rearrange(one_degree_shs, 'n rgb shs_num -> n shs_num rgb')
#     shs_feat[:, 0:3] = one_degree_shs

#     two_degree_shs = shs_feat[:, 3:8]
#     two_degree_shs = einops.rearrange(two_degree_shs, 'n shs_num rgb -> n rgb shs_num')
#     two_degree_shs = einsum(
#             D2,
#             two_degree_shs,
#             "... i j, ... j -> ... i",
#         )
#     two_degree_shs = einops.rearrange(two_degree_shs, 'n rgb shs_num -> n shs_num rgb')
#     shs_feat[:, 3:8] = two_degree_shs

#     three_degree_shs = shs_feat[:, 8:15]
#     three_degree_shs = einops.rearrange(three_degree_shs, 'n shs_num rgb -> n rgb shs_num')
#     three_degree_shs = einsum(
#             D3,
#             three_degree_shs,
#             "... i j, ... j -> ... i",
#         )
#     three_degree_shs = einops.rearrange(three_degree_shs, 'n rgb shs_num -> n shs_num rgb')
#     shs_feat[:, 8:15] = three_degree_shs

#     return shs_feat

def transform_shs(shs_feat, rotation_matrix):
    P = torch.tensor([[0, 0, 1], [1, 0, 0], [0, 1, 0]],dtype=torch.float32,device=rotation_matrix.device) 
    permuted_rotation_matrix = torch.linalg.inv(P) @ rotation_matrix @ P
    rotation_angles = o3._rotation.matrix_to_angles(permuted_rotation_matrix)
    #rotation_angles = pytorch3d.transforms.matrix_to_euler_angles(permuted_rotation_matrix,'YXY')

    return transform_shs_euler(shs_feat, rotation_angles)

def transform_shs_euler(shs_feat, rotation_angles):
    #TODO I harry modified these function libraries to support cuda
    D1 = o3.wigner_D(1, rotation_angles[0], - rotation_angles[1], rotation_angles[2])
    D2 = o3.wigner_D(2, rotation_angles[0], - rotation_angles[1], rotation_angles[2])
    D3 = o3.wigner_D(3, rotation_angles[0], - rotation_angles[1], rotation_angles[2])

    one_degree_shs = shs_feat[:, 0:3]
    one_degree_shs = einops.rearrange(one_degree_shs, 'n shs_num rgb -> n rgb shs_num')
    one_degree_shs = einsum(
            D1,
            one_degree_shs,
            "... i j, ... j -> ... i",
        )
    one_degree_shs = einops.rearrange(one_degree_shs, 'n rgb shs_num -> n shs_num rgb')
    shs_feat[:, 0:3] = one_degree_shs

    two_degree_shs = shs_feat[:, 3:8]
    two_degree_shs = einops.rearrange(two_degree_shs, 'n shs_num rgb -> n rgb shs_num')
    two_degree_shs = einsum(
            D2,
            two_degree_shs,
            "... i j, ... j -> ... i",
        )
    two_degree_shs = einops.rearrange(two_degree_shs, 'n rgb shs_num -> n shs_num rgb')
    shs_feat[:, 3:8] = two_degree_shs

    three_degree_shs = shs_feat[:, 8:15]
    three_degree_shs = einops.rearrange(three_degree_shs, 'n shs_num rgb -> n rgb shs_num')
    three_degree_shs = einsum(
            D3,
            three_degree_shs,
            "... i j, ... j -> ... i",
        )
    three_degree_shs = einops.rearrange(three_degree_shs, 'n rgb shs_num -> n shs_num rgb')
    shs_feat[:, 8:15] = three_degree_shs

    return shs_feat


# def rotate_splat(args,dataset,rotation_matrix):
#     with torch.no_grad():
#         loaded_iter = args.iteration
#         if args.iteration==-1:
#             loaded_iter = searchForMaxIteration(os.path.join(dataset.model_path, "point_cloud"))
#         rotated_output_path = os.path.join(dataset.model_path.replace("render_out","render_out_rot"),"point_cloud",f"iteration_{loaded_iter}", "point_cloud.ply")
        
#         model = GaussianModel(dataset.sh_degree)
#         Scene(dataset, model, load_iteration=args.iteration, shuffle=False,only_test=True)

#         wigner_D_rotated_extra_shs = transform_shs(model.get_features[:, 1:, :].clone().cpu(), rotation_matrix.cpu())

#         wigner_D_rotated_shs = model.get_features.clone().cpu()
#         wigner_D_rotated_shs[:, 1:, :] = wigner_D_rotated_extra_shs

#         rotated_xyz = model.get_xyz @ torch.tensor(so3.as_matrix().T, device=model.get_xyz.device, dtype=torch.float)
#         rotated_rotations = torch.nn.functional.normalize(GaussianTransformUtils.quat_multiply(
#             model.get_rotation, 
#             torch.tensor(so3.as_quaternion_xyzw()[[3, 0, 1, 2]], device=rotated_xyz.device, dtype=torch.float),
#         ))

#         model._xyz = rotated_xyz 
#         model._rotation = rotated_rotations
#         model.save_ply(rotated_output_path)

def rotate_splat(model, scipy_rot):
    rotation_matrix = torch.tensor(scipy_rot.as_matrix(), dtype=torch.float, device="cuda")
    
    wigner_D_rotated_extra_shs = transform_shs(model.get_features[:, 1:, :].clone().cpu(), rotation_matrix.cpu())

    #wigner_D_rotated_shs = model.get_features.clone().cpu()
    #wigner_D_rotated_shs[:, 1:, :] = wigner_D_rotated_extra_shs

    rotated_xyz = model.get_xyz @ torch.tensor(scipy_rot.as_matrix().T, device=model.get_xyz.device, dtype=torch.float)
    rotated_rotations = torch.nn.functional.normalize(GaussianTransformUtils.quat_multiply(
        model.get_rotation, 
        torch.tensor(scipy_rot.as_quat()[[3, 0, 1, 2]], device=rotated_xyz.device, dtype=torch.float),
    ))

    model._xyz = rotated_xyz 
    model._rotation = rotated_rotations
    model._features_rest = wigner_D_rotated_extra_shs.cuda()

def rotate_splat_cuda(model, rotation_matrix, harmonic=True):   
    if harmonic: 
        wigner_D_rotated_extra_shs = transform_shs(model.get_features[:, 1:, :].clone(), rotation_matrix)

    #wigner_D_rotated_shs = model.get_features.clone().cpu()
    #wigner_D_rotated_shs[:, 1:, :] = wigner_D_rotated_extra_shs

    rotation_quat = pytorch3d.transforms.matrix_to_quaternion(rotation_matrix)

    rotated_xyz = model.get_xyz @ rotation_matrix.T
    rotated_rotations = torch.nn.functional.normalize(GaussianTransformUtils.quat_multiply(
        model.get_rotation, 
        #torch.tensor(scipy_rot.as_quat()[[3, 0, 1, 2]], device=rotated_xyz.device, dtype=torch.float),
        rotation_quat
    ))

    model._xyz = rotated_xyz 
    model._rotation = rotated_rotations

    if harmonic:
        model._features_rest = wigner_D_rotated_extra_shs

def rotate_splat_cuda_angle(model, rotation_angles, harmonic=True):    
    if harmonic:
        wigner_D_rotated_extra_shs = transform_shs_euler(model.get_features[:, 1:, :].clone(), rotation_angles)

    #wigner_D_rotated_shs = model.get_features.clone().cpu()
    #wigner_D_rotated_shs[:, 1:, :] = wigner_D_rotated_extra_shs

    permuted_rotation_matrix = o3._rotation.angles_to_matrix(rotation_angles[0], rotation_angles[1], rotation_angles[2])
    P = torch.tensor([[0, 0, 1], [1, 0, 0], [0, 1, 0]],dtype=torch.float32,device=rotation_angles.device) 
    rotation_matrix = P @ permuted_rotation_matrix @ torch.linalg.inv(P)

    rotation_quat = pytorch3d.transforms.matrix_to_quaternion(rotation_matrix)

    rotated_xyz = model.get_xyz @ rotation_matrix.T
    rotated_rotations = torch.nn.functional.normalize(GaussianTransformUtils.quat_multiply(
        model.get_rotation, 
        #torch.tensor(scipy_rot.as_quat()[[3, 0, 1, 2]], device=rotated_xyz.device, dtype=torch.float),
        rotation_quat
    ))

    model._xyz = rotated_xyz 
    model._rotation = rotated_rotations

    if harmonic:
        model._features_rest = wigner_D_rotated_extra_shs

# if __name__ == "__main__":

#     parser = ArgumentParser(description="Testing script parameters")
#     model = ModelParams(parser, sentinel=True)
#     pipeline = PipelineParams(parser)
#     parser.add_argument("--iteration", default=-1, type=int)
#     args = get_combined_args(parser)

#     print("rotating " + args.model_path)

#     x_ang, y_ang, z_ang = (90, 180, 0) #rotation angles in degrees (x,y,z)
#     roll = math.radians(x_ang)   # Rotation around the x-axis
#     pitch = math.radians(y_ang)  # Rotation around the y-axis
#     yaw = math.radians(z_ang)    # Rotation around the z-axis

#     so3 = vst.SO3.from_rpy_radians(roll, pitch, yaw)
#     rotation_matrix = torch.tensor(so3.as_matrix(), dtype=torch.float, device="cuda")

#     rotate_splat(args,model.extract(args),rotation_matrix)