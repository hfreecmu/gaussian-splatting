import sys

import torch
from scene import Scene
import os
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
import torch.nn.functional as F
import torchvision.transforms.functional as func
from seg_utils import conv2d_matrix, compute_ratios, update
import open3d

def project_to_2d(viewpoint_camera, points3D):
    full_matrix = viewpoint_camera.full_proj_transform  # w2c @ K 
    # project to image plane
    if points3D.shape[-1] != 4:
        points3D = F.pad(input=points3D, pad=(0, 1), mode='constant', value=1)
    p_hom = (points3D @ full_matrix).transpose(0, 1)  # N, 4 -> 4, N   -1 ~ 1
    p_w = 1.0 / (p_hom[-1, :] + 0.0000001)
    p_proj = p_hom[:3, :] * p_w

    h = viewpoint_camera.image_height
    w = viewpoint_camera.image_width

    point_image = 0.5 * ((p_proj[:2] + 1) * torch.tensor([w, h]).unsqueeze(-1).to(p_proj.device) - 1) # image plane
    point_image = point_image.detach().clone()
    point_image = torch.round(point_image.transpose(0, 1))

    return point_image

def mask_inverse(xyz, viewpoint_camera, sam_mask):
    w2c_matrix = viewpoint_camera.world_view_transform
    # project to camera space
    xyz = F.pad(input=xyz, pad=(0, 1), mode='constant', value=1)
    p_view = (xyz @ w2c_matrix[:, :3]).transpose(0, 1)  # N, 3 -> 3, N
    depth = p_view[-1, :].detach().clone()
    valid_depth = depth >= 0

    h = viewpoint_camera.image_height
    w = viewpoint_camera.image_width
    

    if sam_mask.shape[0] != h or sam_mask.shape[1] != w:
        sam_mask = func.resize(sam_mask.unsqueeze(0), (h, w), antialias=True).squeeze(0).long()
    else:
        sam_mask = sam_mask.long()

    point_image = project_to_2d(viewpoint_camera, xyz)
    point_image = point_image.long()

    # 判断x,y是否在图像范围之内
    valid_x = (point_image[:, 0] >= 0) & (point_image[:, 0] < w)
    valid_y = (point_image[:, 1] >= 0) & (point_image[:, 1] < h)
    valid_mask = valid_x & valid_y & valid_depth
    point_mask = torch.full((point_image.shape[0],), -1).to("cuda")

    point_mask[valid_mask] = sam_mask[point_image[valid_mask, 1], point_image[valid_mask, 0]]
    indices_mask = torch.where(point_mask == 1)[0]

    return point_mask, indices_mask

# I changed from 0.7
def ensemble(multiview_masks, threshold=0.8):
    # threshold = 0.7
    multiview_masks = torch.cat(multiview_masks, dim=1)
    vote_labels,_ = torch.mode(multiview_masks, dim=1)
    # # select points with score > threshold 
    matches = torch.eq(multiview_masks, vote_labels.unsqueeze(1))
    ratios = torch.sum(matches, dim=1) / multiview_masks.shape[1]
    ratios_mask = ratios > threshold
    labels_mask = (vote_labels == 1) & ratios_mask
    indices_mask = torch.where(labels_mask)[0].detach().cpu()

    return vote_labels, indices_mask

def gaussian_decomp(gaussians, viewpoint_camera, input_mask, indices_mask):
    xyz = gaussians.get_xyz
    point_image = project_to_2d(viewpoint_camera, xyz)

    conv2d = conv2d_matrix(gaussians, viewpoint_camera, indices_mask, device="cuda")
    height = viewpoint_camera.image_height
    width = viewpoint_camera.image_width
    index_in_all, ratios, dir_vector = compute_ratios(conv2d, point_image, indices_mask, input_mask, height, width)

    decomp_gaussians = update(gaussians, viewpoint_camera, index_in_all, ratios, dir_vector)

    return decomp_gaussians

def segment(dataset, iteration, pipeline, output_dir, sor_filter):
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)
    gaussians.save_ply(os.path.join(output_dir, "mask.ply"))
    return

    xyz = gaussians.get_xyz
    cameras = scene.getTrainCameras()

    multiview_masks = []
    #sam_masks = []
    for i, view in enumerate(cameras):
        # sam_mask = torch.any(view.original_image > 0, dim=0)
        sam_mask = (view.object_mask.cuda() > 0).squeeze(0)
        sam_mask = sam_mask.long()
        #sam_masks.append(sam_mask)

        point_mask, indices_mask = mask_inverse(xyz, view, sam_mask)
        multiview_masks.append(point_mask.unsqueeze(-1))

        # if i % 40 == 0:
        #     gaussians = gaussian_decomp(gaussians, view, sam_mask, indices_mask)

    _, final_mask = ensemble(multiview_masks)

    # for i, view in enumerate(cameras):
    #     input_mask = sam_masks[i]
    #     if i % 40 == 0:
    #         gaussians = gaussian_decomp(gaussians, view, input_mask, final_mask.to('cuda'))

    gaussians._xyz = gaussians._xyz[final_mask]
    gaussians._features_dc = gaussians._features_dc[final_mask]
    gaussians._features_rest = gaussians._features_rest[final_mask]
    gaussians._scaling = gaussians._scaling[final_mask]
    gaussians._rotation = gaussians._rotation[final_mask]
    gaussians._opacity = gaussians._opacity[final_mask]

    scales = gaussians.get_scaling
    rad = scales.max(dim=1)[0]
    valid_inds = torch.where(torch.abs(rad - rad.mean()) < 3*rad.std())

    gaussians._xyz = gaussians._xyz[valid_inds]
    gaussians._features_dc = gaussians._features_dc[valid_inds]
    gaussians._features_rest = gaussians._features_rest[valid_inds]
    gaussians._scaling = gaussians._scaling[valid_inds]
    gaussians._rotation = gaussians._rotation[valid_inds]
    gaussians._opacity = gaussians._opacity[valid_inds]

    opacities = gaussians.get_opacity[:, 0]
    valid_inds = torch.where(opacities > 0.025)

    gaussians._xyz = gaussians._xyz[valid_inds]
    gaussians._features_dc = gaussians._features_dc[valid_inds]
    gaussians._features_rest = gaussians._features_rest[valid_inds]
    gaussians._scaling = gaussians._scaling[valid_inds]
    gaussians._rotation = gaussians._rotation[valid_inds]
    gaussians._opacity = gaussians._opacity[valid_inds]

    if sor_filter:
        print('running sor filter')
        pts = gaussians.get_xyz.detach().cpu().numpy()
        pcd = open3d.geometry.PointCloud()
        pcd.points = open3d.utility.Vector3dVector(pts)
        _, valid_inds = pcd.remove_statistical_outlier(nb_neighbors=5, std_ratio=8.0)

        gaussians._xyz = gaussians._xyz[valid_inds]
        gaussians._features_dc = gaussians._features_dc[valid_inds]
        gaussians._features_rest = gaussians._features_rest[valid_inds]
        gaussians._scaling = gaussians._scaling[valid_inds]
        gaussians._rotation = gaussians._rotation[valid_inds]
        gaussians._opacity = gaussians._opacity[valid_inds]


    gaussians.save_ply(os.path.join(output_dir, "mask.ply"))

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--sor_filter', action='store_true')
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    with torch.no_grad():
        segment(model.extract(args), args.iteration, pipeline.extract(args),
                args.output_dir, args.sor_filter)