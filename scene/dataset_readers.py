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
    read_extrinsics_binary, read_intrinsics_binary, read_points3D_binary, read_points3D_text
from utils.graphics_utils import getWorld2View2, focal2fov, fov2focal
import numpy as np
import json
from pathlib import Path
from plyfile import PlyData, PlyElement
from utils.sh_utils import SH2RGB
from scene.gaussian_model import BasicPointCloud
import copy

import trimesh

class CameraInfo(NamedTuple):
    uid: int
    R: np.array
    T: np.array
    FovY: np.array
    FovX: np.array
    cx: np.array
    cy: np.array
    image: np.array
    object_mask: np.array
    human_mask: np.array
    inv_depth: np.array
    image_path: str
    image_name: str
    width: int
    height: int

class SceneInfo(NamedTuple):
    point_cloud: BasicPointCloud
    train_cameras: list
    test_cameras: list
    nerf_normalization: dict
    ply_path: str

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

def readColmapCameras(cam_extrinsics, cam_intrinsics, images_folder, 
                      object_masks_folder, human_masks_folder, inv_depths_folder,
                      max_train_images,
                      ):
    cam_infos = []
    # print('HARRY WARNING YOU ARE RESTRICTING DEMOS')
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
            cx = intr.params[1]
            cy = intr.params[2]
        elif intr.model=="PINHOLE":
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[1]
            FovY = focal2fov(focal_length_y, height)
            FovX = focal2fov(focal_length_x, width)
            cx = intr.params[2]
            cy = intr.params[3]
        elif intr.model=="OPENCV":
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[1]
            FovY = focal2fov(focal_length_y, height)
            FovX = focal2fov(focal_length_x, width)
            cx = intr.params[2]
            cy = intr.params[3]
        else:
            assert False, "Colmap camera model not handled: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras) supported!"

        cx = (cx - width / 2) / width * 2
        cy = (cy - height / 2) / height * 2

        # image_path = os.path.join(images_folder, os.path.basename(extr.name))
        image_path = os.path.join(images_folder, extr.name)
        # if not 'demo' in image_path:
        #     continue
        image_name = os.path.basename(image_path).split(".")[0]
        image = copy.deepcopy(Image.open(image_path))

        object_mask = None
        if object_masks_folder != '':
            obj_mask_path = os.path.join(object_masks_folder, image_name + '.png')
            if os.path.exists(obj_mask_path):
                object_mask = copy.deepcopy(Image.open(obj_mask_path))

        human_mask = None
        if human_masks_folder != '':
            human_mask_path = os.path.join(human_masks_folder, image_name + '.png')
            if os.path.exists(human_mask_path):
                human_mask = copy.deepcopy(Image.open(human_mask_path))
                human_mask = np.array(human_mask)
                human_mask[human_mask > 0] = 255
                human_mask = Image.fromarray(human_mask)

        # not support inv depth right now
        inv_depth = None
        # if inv_depths_folder != '':
        #     inv_depth_path = os.path.join(inv_depths_folder, image_name + '.npy')
        #     if os.path.exists(inv_depth_path):
        #         inv_depth = np.load(inv_depth_path).astype(np.float32)

        cam_info = CameraInfo(uid=uid, R=R, T=T, FovY=FovY, FovX=FovX, cx=cx, cy=cy,
                              image=image, 
                              object_mask=object_mask, human_mask=human_mask, inv_depth=inv_depth,
                              image_path=image_path, image_name=image_name, width=width, height=height)
        cam_infos.append(cam_info)
    
    if max_train_images is not None and len(cam_infos) > max_train_images:
        raise RuntimeError('not yet')
        rand_inds = np.random.choice(len(cam_infos), size=max_train_images, replace=False)
        cam_infos = [cam_infos[ind] for ind in rand_inds]
        
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

# changing to 16 for now
# back to 8
def readColmapSceneInfo(path, images, object_masks, human_masks, inv_depths,
                        eval, llffhold=8, max_train_images=None):
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
    # try:
    #     cameras_extrinsic_file = os.path.join(path, "sparse/0_refine_out", "images.bin")
    #     cameras_intrinsic_file = os.path.join(path, "sparse/0_refine_out", "cameras.bin")
    #     cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
    #     cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
    # except:
    #     cameras_extrinsic_file = os.path.join(path, "sparse/0_refine_out", "images.txt")
    #     cameras_intrinsic_file = os.path.join(path, "sparse/0_refine_out", "cameras.txt")
    #     cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file)
    #     cam_intrinsics = read_intrinsics_text(cameras_intrinsic_file)

    reading_dir = "images" if images == None else images
    
    # object_masks_dir = 'mask_obj' if object_masks == None else object_masks
    # human_masks_dir = 'mask_human' if human_masks == None else human_masks
    # inv_depths_dir = 'inv_depth' if inv_depths == None else inv_depths
    cam_infos_unsorted = readColmapCameras(cam_extrinsics=cam_extrinsics, cam_intrinsics=cam_intrinsics, 
                                           images_folder=os.path.join(path, reading_dir),
                                        #    object_masks_folder=os.path.join(path, object_masks_dir),
                                        #    human_masks_folder=os.path.join(path, human_masks_dir),
                                        #    inv_depths_folder = os.path.join(path, inv_depths_dir),
                                           object_masks_folder=object_masks,
                                           human_masks_folder=human_masks,
                                           inv_depths_folder=inv_depths,
                                           max_train_images=max_train_images,
                                           )
    cam_infos = sorted(cam_infos_unsorted.copy(), key = lambda x : x.image_name)

    if eval:
        #changing to mod -1 so does not take first 4
        # train_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold != llffhold - 1 or 'gauss' not in c.image_name]
        # test_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold == llffhold - 1 and 'gauss' in c.image_name]
        train_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold != llffhold - 1]
        test_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold == llffhold - 1]
    else:
        train_cam_infos = cam_infos
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

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

    # ply_path = os.path.join(path, "sparse/0_refine_out/points3D.ply")
    # bin_path = os.path.join(path, "sparse/0_refine_out/points3D.bin")
    # txt_path = os.path.join(path, "sparse/0_refine_out/points3D.txt")
    # if not os.path.exists(ply_path):
    #     print("Converting point3d.bin to .ply, will happen only the first time you open the scene.")
    #     try:
    #         xyz, rgb, _ = read_points3D_binary(bin_path)
    #     except:
    #         xyz, rgb, _ = read_points3D_text(txt_path)
    #     storePly(ply_path, xyz, rgb)
    # try:
    #     pcd = fetchPly(ply_path)
    # except:
    #     pcd = None

    #ply_path = os.path.join(path, "sparse/0/trimmed.ply")
    # ply_path = os.path.join(path, "dense_refine/points3D_multipleview.ply")
    #ply_path = os.path.join(path, "dense/apple.ply")
    #pcd = fetchPly(ply_path)

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info

def readBundleCameras(extrinsics_dir, K_path, images_dir, masks_dir):
    raise RuntimeError('not used?')
    intr = np.loadtxt(K_path)
    uid = 0
    cam_infos = []
    for filename in os.listdir(images_dir):
        if not (filename.endswith('.jpg') or filename.endswith('.png')):
            continue
        basename = filename.replace('.jpg', '').replace('.png', '')

        extrinsics_path = os.path.join(extrinsics_dir, basename + '.txt')
        extr = np.loadtxt(extrinsics_path)

        R = np.transpose(extr[0:3, 0:3])
        T = np.array(extr[0:3, 3])

        image_path = os.path.join(images_dir, filename)
        mask_path = os.path.join(masks_dir, basename + '.png')

        image_name = os.path.basename(image_path).split(".")[0]
        image = copy.deepcopy(Image.open(image_path))
        width, height = image.size
        object_mask = np.array(copy.deepcopy(Image.open(mask_path)))
        object_mask = object_mask.astype(float) / 255.0

        # cam stuff
        fx = intr[0, 0]
        fy = intr[1, 1]
        cx = intr[0, 2]
        cy = intr[1, 2]

        FovY = focal2fov(fy, height)
        FovX = focal2fov(fx, width)
        cx = (cx - width / 2) / width * 2
        cy = (cy - height / 2) / height * 2
        #

        cam_info = CameraInfo(uid=uid, R=R, T=T, FovY=FovY, FovX=FovX, cx=cx, cy=cy,
                              image=image, object_mask=object_mask,
                              image_path=image_path, image_name=image_name, width=width, height=height)
        
        cam_infos.append(cam_info)

    sys.stdout.write('\n')
    return cam_infos

def readBundleSceneInfo(path, images_dep, object_masks_dep, eval_dep, llffhold_dep=8, 
                        max_train_images_dep=None):
    raise RuntimeError('not used?')
    extrinsics_dir = os.path.join(path, 'ob_in_cam')
    masks_dir = os.path.join(path, 'masks')
    images_dir = os.path.join(path, 'images')
    K_path = os.path.join(path, 'cam_K.txt')

    cam_infos_unsorted = readBundleCameras(extrinsics_dir, K_path,
                                           images_dir, masks_dir)
    cam_infos = sorted(cam_infos_unsorted.copy(), key = lambda x : x.image_name)

    train_cam_infos = cam_infos
    test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    mesh_path = os.path.join(path, 'textured_mesh.obj')
    mesh = trimesh.load(mesh_path, process=False)

    ply_path = os.path.join(path, 'points3D.ply')

    if not os.path.exists(ply_path):
        xyz = np.array(mesh.vertices)
        rgb = np.array(mesh.visual.to_color().vertex_colors)[:, 0:3]
        storePly(ply_path, xyz, rgb)

    pcd = fetchPly(ply_path)

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)

    return scene_info

def readCamerasFromTransforms(path, transformsfile, white_background, extension=".png"):
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

            cam_infos.append(CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                            image_path=image_path, image_name=image_name, width=image.size[0], height=image.size[1]))
            
    return cam_infos

def readNerfSyntheticInfo(path, white_background, eval, extension=".png"):
    print("Reading Training Transforms")
    train_cam_infos = readCamerasFromTransforms(path, "transforms_train.json", white_background, extension)
    print("Reading Test Transforms")
    test_cam_infos = readCamerasFromTransforms(path, "transforms_test.json", white_background, extension)
    
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
                           ply_path=ply_path)
    return scene_info

sceneLoadTypeCallbacks = {
    "Colmap": readColmapSceneInfo,
    #"BundleSdf": readBundleSceneInfo,
    "Blender" : readNerfSyntheticInfo
}