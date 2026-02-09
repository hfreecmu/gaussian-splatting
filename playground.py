import json
import torch
import numpy as np
import cv2
from gsplat.rendering import rasterization
from pathlib import Path
from sklearn.decomposition import PCA
from torchvision.transforms.functional import resize

from scene.gaussian_dino_model import GaussianDinoModel, GAUSSIAN_DINO_DIM
from dino_utils.dino_dataloader import DinoDataloader
from argparse import ArgumentParser
from arguments import PipelineParams
from gaussian_renderer import my_render

def read_json(path):
    with open(path) as f:
        data = json.load(f)
    return data

def splat_to_image_color(tensor):
    img = tensor.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to("cpu", torch.uint8).numpy()
    return img

SPLAT_PATH = '/home/hfreeman/harry_ws/repos/pruner_track/datasets/SCENES/oven_scene/splat/point_cloud/iteration_7000/point_cloud.ply'
CAMERAS_PATH = '/home/hfreeman/harry_ws/repos/pruner_track/datasets/SCENES/oven_scene/splat/cameras.json'
CACHE_PATH = '/home/hfreeman/harry_ws/repos/pruner_track/datasets/SCENES/oven_scene/splat/dino/dino.npy'
IMAGE_IND = 10

gaussians = GaussianDinoModel(3)
gaussians.load_ply(SPLAT_PATH)

cameras = read_json(CAMERAS_PATH)
camera = cameras[IMAGE_IND]

parser = ArgumentParser()
pipeline_par = PipelineParams(parser)
args, _ = parser.parse_known_args()
pipeline = pipeline_par.extract(args)
bg_color = [0, 0, 0]
background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

fx = camera['fx']
fy = camera['fy']
cx = camera['cx']
cy = camera['cy']

height = camera['height']
width = camera['width']

intrinsics = [fx, fy, cx, cy]
dims = [height, width]

rot_mat = np.array(camera['rotation'])
trans = np.array(camera['position'])
M = np.eye(4)
M[0:3, 0:3] = rot_mat
M[0:3, 3] = trans
M = np.linalg.inv(M)
rot_mat = M[0:3, 0:3]
trans = M[0:3, 3]

with torch.no_grad():
    res_pkg = my_render(
        gaussians,
        pipeline,
        background,
        intrinsics,
        dims,
        rot_mat.T,
        trans,
    )

render = res_pkg['render']
res_im = splat_to_image_color(render)

cv2.imshow('test', cv2.cvtColor(res_im, cv2.COLOR_RGB2BGR))
# cv2.waitKey(0)

dino_feats_col = gaussians.get_dino_feats
viewmat = torch.FloatTensor(M).cuda()[None]

K = np.array([[fx, 0, cx],
              [0, fy, cy],
               [0, 0, 1.0]])
K = torch.FloatTensor(K).cuda()

dino_K = K[None].clone()
dino_h, dino_w = height, width

with torch.no_grad():
    dino_feats, dino_alpha, _ = rasterization(
        means=gaussians.get_xyz, # in not training dont' detach
        quats=gaussians.get_rotation.detach(),
        scales=gaussians.get_scaling.detach(),
        opacities=gaussians.get_opacity[:, 0].detach(),
        colors=dino_feats_col,
        viewmats=viewmat,  # [1, 4, 4]
        Ks=dino_K,
        width=dino_w,
        height=dino_h,
        packed=False,
        near_plane=0.01,
        far_plane=1e10,
        render_mode="RGB",
        sparse_grad=False,
        absgrad=False,
        rasterize_mode="classic",
        tile_size = 10
    )

feat_shape = dino_feats.shape
dino_feats = torch.where(dino_alpha > 0, dino_feats / dino_alpha.detach(), torch.zeros(GAUSSIAN_DINO_DIM, device='cuda'))
nn_inputs = dino_feats.view(-1, GAUSSIAN_DINO_DIM)
dino_feats = gaussians.dino_nn(nn_inputs).view(*feat_shape[:-1],-1).squeeze(0)

# for not training
dino_feats[dino_alpha.squeeze(-1).squeeze(0) < 0.8] = 0

alpha_hw = dino_alpha.squeeze(0).squeeze(-1).detach().cpu().numpy()
valid = alpha_hw >= 0.8


data_loader = DinoDataloader(
    image_list=None,
    device='cuda',
    cfg={"image_shape": [height, width]},
    cache_path=Path(CACHE_PATH)

)
gt_dino = data_loader.get_full_img_feats(IMAGE_IND)

# TODO proper calculate like robot see robot do to compare
# gt_dino = resize(gt_dino.permute(2,0,1), (dino_feats.shape[0],dino_feats.shape[1])).permute(1,2,0)
# dino_feats = resize(dino_feats.permute(2,0,1), (gt_dino.shape[0],gt_dino.shape[1])).permute(1,2,0)

feat = dino_feats.detach().float().cpu().numpy() 
gt_feat   = gt_dino.detach().float().cpu().numpy()

Hp, Wp, Cp = dino_feats.shape
Hg, Wg, Cg = gt_dino.shape

gt_pca = PCA(3)
gt_pca.fit(gt_feat.reshape((Hg*Wg, Cg)))
gt_feat_pca = gt_pca.transform(gt_feat.reshape((Hg*Wg, Cg))).reshape(Hg, Wg, 3)

gt_feat_pca = (gt_feat_pca - gt_feat_pca.min()) / (gt_feat_pca.max() - gt_feat_pca.min())
gt_feat_pca = (gt_feat_pca*255).astype(np.uint8)
cv2.imshow('gt_feat_pca', cv2.cvtColor(gt_feat_pca, cv2.COLOR_RGB2BGR))

pred_pca = PCA(3)
pred_pca.fit(feat.reshape((Hp*Wp, Cp))[valid.flatten()])
# pred_pca.fit(feat.reshape((Hp*Wp, Cp)))
pred_feat_pca = pred_pca.transform(feat.reshape((Hp*Wp, Cp))).reshape(Hp, Wp, 3)
pred_feat_pca = (pred_feat_pca - pred_feat_pca.min()) / (pred_feat_pca.max() - pred_feat_pca.min())
pred_feat_pca = (pred_feat_pca*255).astype(np.uint8)
cv2.imshow('pred_feat_pca', cv2.cvtColor(pred_feat_pca, cv2.COLOR_RGB2BGR))

cv2.waitKey(0)
