import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
import os
import pyquaternion
from pyquaternion import Quaternion
from PIL import Image
import argparse
from tqdm import tqdm



from segment_anything import sam_model_registry, SamPredictor
from sam.automatic_mask_generator import SamAutomaticMaskGenerator
import mmcv

def parse_args():
    parser = argparse.ArgumentParser(description='sam mask arg parser')
    parser.add_argument(
        '--data_root',
        type=str,
        default='./project/data/nuscenes/',
        help='specify the root path of dataset')
    parser.add_argument(
        '--save_path',
        type=str,
        default='/cluster/scratch/scharyyev/thesis/sam_points',
        required=False,
        help='where to save the points after grouping')
    parser.add_argument(
        '--ann_file',
        type=str,
        default='/cluster/scratch/scharyyev/thesis/sam_points',
        required=False,
        help='filename')
    
    args = parser.parse_args()
    return args


def proj_points_camera(points, lidar2img, dims):
    """Project points to image pixels. 

    Args:
        points (np.array): Points
        lidar2img (np.array): Camera matrix
        dims (list): Image dimensions
        
    Returns:
        cam_points (np.array): Pixel positions
        valid_mask (np.array): Mask indicating which points are visible in image

    """   
    lidar2img = torch.from_numpy(lidar2img)
    points = torch.from_numpy(points)

    hom_points = torch.cat((points, torch.ones_like(points[..., :1])), -1)
    cam_points = torch.einsum("ij, hj->hi", lidar2img, hom_points)
    cam_points = cam_points[..., 0:2] / torch.maximum(
            cam_points[..., 2:3], torch.ones_like(cam_points[..., 2:3]) * 1e-5)
    
    # depth_coords = this_coor[:, :2].type(torch.long)
    valid_mask = ((cam_points[:, 1] < dims[0])
                & (cam_points[:, 0] < dims[1])
                & (cam_points[:, 1] >= 0)
                & (cam_points[:, 0] >= 0))
    
    cam_points = cam_points[valid_mask]
    cam_points[:, [0, 1]] = cam_points[:, [1, 0]]
    
    return cam_points.numpy(), valid_mask.numpy()

    


if __name__ == "__main__":
    args = parse_args()
    sam_checkpoint = "/cluster/scratch/scharyyev/thesis/sam/sam_vit_h_4b8939.pth"
    model_type = "vit_h"

    device = "cuda"

    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)

    mask_generator = SamAutomaticMaskGenerator(sam)
    
    
    # data
    data_root = args.data_root
    save_path = args.save_path
    ann_file = args.ann_file
    vis_path = "/cluster/scratch/scharyyev/thesis/unimae/vis/sam"
    file_client_args = dict(backend='disk')

    data = mmcv.load(os.path.join(data_root, ann_file))
    data_infos = list(sorted(data['infos'], key=lambda e: e['timestamp']))
    for index, info in tqdm(enumerate(data_infos)):
        pts_filename=os.path.join(data_root, info['lidar_path'])
        
        # ego to global transform
        ego2global = np.eye(4).astype(np.float32)
        ego2global[:3, :3] = Quaternion(info["ego2global_rotation"]).rotation_matrix
        ego2global[:3, 3] = info["ego2global_translation"]

        # lidar to ego transform
        lidar2ego = np.eye(4).astype(np.float32)
        lidar2ego[:3, :3] = Quaternion(info["lidar2ego_rotation"]).rotation_matrix
        lidar2ego[:3, 3] = info["lidar2ego_translation"]
        
        # points = np.fromfile(pts_filename, dtype=np.float32)
        file_client = mmcv.FileClient(**file_client_args)
        pts_bytes = file_client.get(pts_filename)
        points = np.frombuffer(pts_bytes, dtype=np.float32)
        points = points.reshape(-1, 5).copy()
        labels = np.zeros((points.shape[0], 6), dtype = np.float32)
        cam_num = 0

        for _, camera_info in info["cams"].items():
            image_path = os.path.join(data_root, camera_info["data_path"])
            # lidar to camera transform
            lidar2camera_r = np.linalg.inv(camera_info["sensor2lidar_rotation"])
            lidar2camera_t = (
                camera_info["sensor2lidar_translation"] @ lidar2camera_r.T
            )
            lidar2camera_rt = np.eye(4).astype(np.float32)
            lidar2camera_rt[:3, :3] = lidar2camera_r.T
            lidar2camera_rt[3, :3] = -lidar2camera_t
            lidar2camera = lidar2camera_rt.T

            # camera intrinsics
            camera_intrinsics = np.eye(4).astype(np.float32)
            camera_intrinsics[:3, :3] = camera_info["camera_intrinsics"]

            # lidar to image transform
            lidar2image = camera_intrinsics @ lidar2camera_rt.T
            
            # camera to ego transform
            camera2ego = np.eye(4).astype(np.float32)
            camera2ego[:3, :3] = Quaternion(
                camera_info["sensor2ego_rotation"]
            ).rotation_matrix
            camera2ego[:3, 3] = camera_info["sensor2ego_translation"]

            # camera to lidar transform
            camera2lidar = np.eye(4).astype(np.float32)
            camera2lidar[:3, :3] = camera_info["sensor2lidar_rotation"]
            camera2lidar[:3, 3] = camera_info["sensor2lidar_translation"]
            
            image = Image.open(image_path)
            image = np.asarray(image)
            cam_points, valid_points = proj_points_camera(points[:, 0:3], lidar2image, (image.shape[0], image.shape[1]))
            cam_points  = cam_points.astype(int)
            #show_proj(image, cam_points, valid_points, os.path.join(vis_path, "proj"), str(index) + "_" + _ + ".png")
            
            masks = mask_generator.generate(image)

            for i, ann in enumerate(masks):
                mask = ann['segmentation']
                labels[valid_points, cam_num] = np.maximum((i+1) * mask[cam_points[:, 0], cam_points[:, 1]], labels[valid_points, cam_num])
            # save_anns(image, masks, os.path.join(vis_path, "images", str(index) + "_" + _ + ".png"))
            cam_num += 1
        

        labeled_points = np.concatenate((points, labels), axis = 1).astype(np.float32)
        labeled_points.tofile(os.path.join(save_path, pts_filename.split("/")[-1]))
        

                
        

def show_anns(anns):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:,:,3] = 0
    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [0.35]])
        img[m] = color_mask
    ax.imshow(img)

def show_mask(mask):
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones((mask.shape[0], mask.shape[1], 4))
    img[:,:,3] = 0
    for ann in range(1, mask.max()):
        m = mask == ann
        color_mask = np.concatenate([np.random.random(3), [0.35]])
        img[m] = color_mask
    ax.imshow(img)


def show_proj(image, points, color, path, name):
    points = points
    # points[:, [0, 1]] = points[:, [1, 0]]
    color = color 
    color = np.linalg.norm(color, axis = 1)
    color = color / color.max()
    os.makedirs(path, exist_ok=True)

    fig, ax = plt.subplots(1, 1)
    
    ax.imshow(image.astype(int))
    ax.scatter(points[:, 0], points[:, 1], c=color, s=5)
    ax.axis('off')

    plt.savefig(os.path.join(path, name))


def save_anns(image, masks, name):
    plt.figure(figsize=(20,20))
    plt.imshow(image)
    show_anns(masks)
    plt.axis('off')
    plt.savefig(name)
 