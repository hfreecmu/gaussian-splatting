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
import torch
from random import randint
from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import render, network_gui
import sys
from scene import Scene
from scene.gaussian_dino_model import GaussianDinoModel, GAUSSIAN_DINO_DIM
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

import numpy as np
from gaussian_alpha_renderer import render as alpha_render

from gsplat.rendering import rasterization
from utils.graphics_utils import fov2focal, c2c_orig

from dino_utils.dino_dataloader import get_img_resolution,MAX_DINO_SIZE, DinoDataloader
from pathlib import Path
from torchvision.transforms.functional import resize
from pytorch3d.ops import knn_points

DINO_RESCALE_FACTOR = 5

def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from):
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianDinoModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, shuffle=False)
    gaussians.training_setup(opt)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    viewpoint_ind_stack = None
    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1

    images = []
    for vc in scene.getTrainCameras():
        images.append(vc.original_image.cuda())
    images = torch.stack(images)

    dino_cache_dir = os.path.join(scene.model_path, 'dino')
    if not os.path.exists(dino_cache_dir):
        os.mkdir(dino_cache_dir)

    dino_data_loader = DinoDataloader(
        image_list = images,
        device = 'cuda',
        cfg={"image_shape": list(images.shape[2:4])},
        cache_path=Path(os.path.join(dino_cache_dir, 'dino.npy'))
    )
    nearest_ids = None

    for iteration in range(first_iter, opt.iterations + 1):        
        if network_gui.conn == None:
            network_gui.try_connect()
        while network_gui.conn != None:
            try:
                net_image_bytes = None
                custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
                if custom_cam != None:
                    net_image = render(custom_cam, gaussians, pipe, background, scaling_modifer)["render"]
                    net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                network_gui.send(net_image_bytes, dataset.source_path)
                if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                    break
            except Exception as e:
                network_gui.conn = None

        iter_start.record()

        gaussians.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Pick a random Camera
        if not viewpoint_ind_stack:
            viewpoint_ind_stack = np.arange(len(scene.getTrainCameras())).tolist()

        viewpoint_ind = viewpoint_ind_stack.pop(randint(0, len(viewpoint_ind_stack)-1))
        viewpoint_cam = scene.getTrainCameras()[viewpoint_ind]
        gt_dino = dino_data_loader.get_full_img_feats(viewpoint_ind)

        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True

        bg = torch.rand((3), device="cuda") if opt.random_background else background

        is_mask = False
        is_inv_depth = False
        gt_image = viewpoint_cam.original_image.cuda()
        
        if viewpoint_cam.object_mask is not None:
            gt_object_mask = viewpoint_cam.object_mask.cuda()
            gt_image = torch.where(gt_object_mask > 0, gt_image, torch.zeros_like(gt_image))
            is_mask = True
        
        if viewpoint_cam.human_mask is not None:
            gt_human_mask = viewpoint_cam.human_mask.cuda()
            valid_pix = 1 - gt_human_mask
        else:
            valid_pix = torch.ones_like(gt_image)

        if viewpoint_cam.inv_depth is not None:
            gt_inv_depth = viewpoint_cam.inv_depth.cuda()
            is_inv_depth = True

        # render_pkg = render(viewpoint_cam, gaussians, pipe, bg)
        render_pkg = alpha_render(viewpoint_cam, gaussians, pipe, bg)
        
        dino_feats_col = gaussians.get_dino_feats
        
        width = viewpoint_cam.image_width
        height = viewpoint_cam.image_height

        fov_x = viewpoint_cam.FoVx
        fov_y = viewpoint_cam.FoVy
        cx = viewpoint_cam.cx
        cy = viewpoint_cam.cy

        fx = fov2focal(fov_x, width)
        fy = fov2focal(fov_y, height)
        cx = c2c_orig(cx, width)
        cy = c2c_orig(cy, height)

        K = np.array([[fx, 0, cx],
                        [0, fy, cy],
                        [0, 0, 1.0]])
        K = torch.FloatTensor(K).cuda()
        viewmat = viewpoint_cam.world_view_transform.T[None]

        h,w = get_img_resolution(height, width)
        # this isn't done for not training
        dino_h,dino_w = DINO_RESCALE_FACTOR*(h//14),DINO_RESCALE_FACTOR*(w//14)

        dino_K = K[None].clone()
        # this isn't done for not training
        downscale = (DINO_RESCALE_FACTOR*MAX_DINO_SIZE/max(height,width))/14
        dino_K[:, :2, :] *= downscale

        dino_feats, dino_alpha, _ = rasterization(
            means=gaussians.get_xyz.detach(), # in not training dont' detach
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

        gt_dino = resize(gt_dino.permute(2,0,1), (dino_feats.shape[0],dino_feats.shape[1])).permute(1,2,0)
        gt_dino = torch.where(dino_alpha.squeeze(0) > 0, gt_dino, torch.zeros_like(gt_dino))
        dino_loss = torch.nn.functional.mse_loss(dino_feats, gt_dino)

        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]

        # Loss
        image = torch.where(valid_pix > 0, image, torch.zeros_like(gt_image))
        gt_image = torch.where(valid_pix > 0, gt_image, torch.zeros_like(gt_image))

        Ll1 = l1_loss(image, gt_image)
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))

        loss += dino_loss

        if iteration > 1000:
            if nearest_ids is None or gaussians.get_xyz.shape[0] != nearest_ids.shape[0]:
                means = gaussians.get_xyz.detach()
                nearest_ids = knn_points(means.unsqueeze(0), means.unsqueeze(0), K=3, return_nn=False)[1].squeeze(0)
            
            #digs does sum
            dino_nn_loss = 0.01 * gaussians.get_dino_feats[nearest_ids].var(dim=1).mean()
            # dino_nn_loss = 0.01 * gaussians.get_dino_feats[nearest_ids].var(dim=1).sum()

            loss += dino_nn_loss

        if is_mask:
            rend_mask = torch.where(valid_pix > 0, render_pkg['alpha'], torch.zeros_like(render_pkg['alpha']))
            gt_mask = torch.where(valid_pix > 0, gt_object_mask, torch.zeros_like(gt_object_mask))
            loss += l1_loss(rend_mask, gt_mask)

        # if is_inv_depth and iteration > 3000:
        #     fov_x = viewpoint_cam.FoVx
        #     fov_y = viewpoint_cam.FoVy
        #     cx = viewpoint_cam.cx
        #     cy = viewpoint_cam.cy

        #     width = viewpoint_cam.image_width
        #     height = viewpoint_cam.image_height

        #     fx = fov2focal(fov_x, width)
        #     fy = fov2focal(fov_y, height)
        #     cx = c2c_orig(cx, width)
        #     cy = c2c_orig(cy, height)

        #     K = np.array([[fx, 0, cx],
        #                   [0, fy, cy],
        #                   [0, 0, 1.0]])
        #     K = torch.FloatTensor(K).cuda()

        #     viewmat = viewpoint_cam.world_view_transform.T

        #     render_colors, _, _ = rasterization(
        #         means=gaussians.get_xyz,
        #         quats=gaussians.get_rotation,
        #         scales=gaussians.get_scaling,
        #         opacities=gaussians.get_opacity[:, 0],
        #         colors=gaussians.get_features,
        #         viewmats=viewmat.unsqueeze(0),
        #         Ks=K.unsqueeze(0),
        #         width=width,
        #         height=height,
        #         render_mode="ED",
        #         sh_degree=dataset.sh_degree,
        #     )

        #     depth = render_colors.squeeze(0).permute(2, 0, 1)
        #     disparity = torch.where(depth > 0, 1 / depth, torch.zeros_like(depth))

        #     min_disparity = disparity.min().detach()
        #     max_disparity = disparity.max().detach()

        #     norm_disparity = (disparity - min_disparity) / (max_disparity - min_disparity)

        #     med_val = norm_disparity.median().detach()
        #     scale_val = torch.abs(norm_disparity - med_val).mean().detach()
        #     shifted_pred = (norm_disparity - med_val) / scale_val
        #     shifted_pred = torch.where(valid_pix > 0, shifted_pred, torch.zeros_like(shifted_pred))

        #     med_gt = gt_inv_depth[valid_pix > 0].median()
        #     scale_gt = torch.abs(gt_inv_depth[valid_pix > 0] - med_gt).mean()
        #     shifted_gt = (gt_inv_depth - med_gt) / scale_gt
        #     shifted_gt = torch.where(valid_pix > 0, shifted_gt, torch.zeros_like(shifted_gt))

        #     raise RuntimeError('not right here')
        #     depth_loss = l1_loss(image, gt_image)
        #     loss += depth_loss
        
        loss.backward()

        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background))
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)

            # Densification
            if iteration < opt.densify_until_iter:
                # Keep track of max radii in image-space for pruning
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold)
                
                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()

            # Optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)

            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")

def prepare_output_and_logger(args):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs)["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    
                    if viewpoint.object_mask is not None:
                        gt_object_mask = torch.clamp(viewpoint.object_mask.to("cuda"), 0.0, 1.0)
                        gt_image = gt_image * gt_object_mask
                        image = image * gt_object_mask

                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])          
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from)

    # All done
    print("\nTraining complete.")
