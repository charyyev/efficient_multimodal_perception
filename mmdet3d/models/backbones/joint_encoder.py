import torch.nn as nn
import torch
import numpy as np
import math
import matplotlib.pyplot as plt
import os

from mmdet.models.builder import BACKBONES
from mmdet3d.models.builder import build_backbone


@BACKBONES.register_module()
class JointEncoder(nn.Module):
    """Encoder for both range image and image modalities
    
    Args:
        lidar_encoder (nn.Module): Encoder for range image
        camera_encoder (nn.Module): Image encoder 
    """
    def __init__(self, 
                 lidar_encoder,
                 camera_encoder,
                 train_cfg = None,
                 test_cfg = None
                 ):
        super(JointEncoder, self).__init__()
        self.lidar_encoder = build_backbone(lidar_encoder)
        self.camera_encoder = build_backbone(camera_encoder)

        self.embed_dims = 192
        
        self.position_encoder = nn.Sequential(
                nn.Linear(3, self.embed_dims*4),
                nn.ReLU(),
                nn.Linear(self.embed_dims*4, self.embed_dims),
            )

    
    def forward(self,
                img,
                range_image,
                img_metas,
                range_points):
        """Compute call for encoder. 

        Args:
            img (torch.tensor): Multi-view images.
            range_img (torch.tensor): Range image.
            img_metas (dict): Image information such as size, intrinsic and extrinsic
            range_points (torch.tensor): Points in the form of range image (each pixel in range image contains x, y, z)
            
        Returns:
            range_features (torch.tensor): Extracted features of range image.
            range_mask (torch.tensor): Masked pixels of range image for MAE
            image_features (torch.tensor): Extrated features of images
            img_mask (torch.tensor): Masked pixels of image for MAE
            range_cam_coors (torch.tensor): projection of range image to image coors, used for MAE reconstruction

        """

        B, N, C, H, W = img.size()
        img = img.view(B * N, C, H, W)

        # features after 2nd layer of encoder
        if self.camera_encoder.mae_cfg is None:
            mid_features = self.camera_encoder.forward1(img)
        else:
            mid_features, img_mask = self.camera_encoder.forward1(img)
        h, w = mid_features.shape[-2:]
        mid_features = mid_features.view(B, N, -1, h, w)

        # mask range image
        range_image, range_mask = self.lidar_encoder.create_masked_input(range_image)
        
        # augment each mmodality with complementary information
        range_features, image_features, range_cam_coors = self.interact(mid_features, range_image, img_metas, range_points)

        # extract range image features
        if self.lidar_encoder.mae_cfg is None:
            range_features, range_mask = self.lidar_encoder(range_features)
        else:    
            range_features, range_mask = self.lidar_encoder([range_features, range_mask])
            range_mask = ~range_mask
        
        # extract image features
        image_features = image_features.view(B * N, -1, h, w)
        if self.camera_encoder.mae_cfg is None:
            image_features = self.camera_encoder.forward2(image_features)
            img_mask = 0
        else:
            image_features, img_mask = self.camera_encoder.forward2([image_features, img_mask])
        image_features = image_features.view(B, N, -1, image_features.shape[-2], image_features.shape[-1])

        return range_features, range_mask, image_features, img_mask, range_cam_coors
        

    def interact(self,
                img_features,
                range_image,
                img_metas,
                range_points):
        """augment each mmodality with complementary information, depth information for images and image feature for range image 

        Args:
            img_features (torch.tensor): Features of images after 2nd layer of encoder 
            range_image (torch.tensor): Range image.
            img_metas (dict): Image information such as size, intrinsic and extrinsic
            range_points (torch.tensor): Points in the form of range image (each pixel in range image contains x, y, z)
            
        Returns:
            torch.tensor: Concatenated image features with range image.
            img_features (torch.tensor): Image features augmented with depth
            range_cam_coors (torch.tensor): projection of range image to image coors, used for MAE reconstruction

        """
        
        resize_dims = img_metas[0]["img_shape"][::-1]
        lidar2imgs = []
        img_augs = []
        for img_meta in img_metas:
            lidar2imgs.append(img_meta['lidar2image'])
            img_augs.append(img_meta["imgs_aug"])

        lidar2imgs = range_points.new_tensor(lidar2imgs)
        
        # project points to image coordinates
        hom_points = torch.cat((range_points, torch.ones_like(range_points[..., :1])), -1)
        cam_points = torch.einsum("bcij, bhwj->bchwi", lidar2imgs, hom_points)
        cam_points = cam_points[..., 0:2] / torch.maximum(
                cam_points[..., 2:3], torch.ones_like(cam_points[..., 2:3]) * 1e-5)
        num_cam = lidar2imgs.shape[1]
        
        # unmaked pixels of range image
        batch_range_mask = range_image > 0
        batch_size = range_points.shape[0]
        
        # pixels that contain no point in range image
        batch_no_point_mask = torch.ones_like(range_image).squeeze(1)
        batch_no_point_mask[(range_points == 0).sum(dim = 3) == 3] = 0
        batch_no_point_mask = batch_no_point_mask.type(torch.bool)

        cam_range_features = torch.zeros(batch_size, img_features.shape[2], range_image.shape[2], range_image.shape[3], device = range_image.device)
        range_cam_coors = torch.zeros(batch_size, num_cam, range_image.shape[2], range_image.shape[3], 2, device = range_image.device) - 1

        # adjust projection with augmentation
        for i in range(batch_size):
            this_coors = cam_points[i]
            range_mask = batch_range_mask[i].squeeze(0)
            no_point_mask = batch_no_point_mask[i]
            no_point_range_mask = range_mask[no_point_mask]
            # range_mask_index = range_mask.squeeze(0).unsqueeze(2).repeat(1, 1, 2)

            resize = [aug["resize"] for aug in img_augs[i]]
            crop = [aug["crop"] for aug in img_augs[i]]
            flip = [aug["flip"] for aug in img_augs[i]]
            
            for cam_it in range(num_cam):
                this_coor = this_coors[cam_it][no_point_mask, :]
                H, W = resize_dims
                this_coor[:, :2] = this_coor[:, :2] * resize[cam_it]
                this_coor[:, 0] -= crop[cam_it][0]
                this_coor[:, 1] -= crop[cam_it][1]
                if flip[cam_it]:
                    this_coor[:, 0] = resize_dims[1] - this_coor[:, 0]

                this_coor[:, 0] -= W / 2.0
                this_coor[:, 1] -= H / 2.0

                h = 0.
                rot_matrix = this_coor.new_tensor([
                    [math.cos(h), math.sin(h)],
                    [-math.sin(h), math.cos(h)],
                ])
                this_coor[:, :2] = torch.matmul(rot_matrix, this_coor[:, :2].T).T

                this_coor[:, 0] += W / 2.0
                this_coor[:, 1] += H / 2.0

                # points that lie in image
                valid_mask = ((this_coor[:, 1] < resize_dims[0])
                            & (this_coor[:, 0] < resize_dims[1])
                            & (this_coor[:, 1] >= 0)
                            & (this_coor[:, 0] >= 0))
                valid_points = range_points[i, no_point_mask, :][valid_mask, :]
                valid_coor = this_coor[valid_mask, :]
                valid_coor[:, [0, 1]] = valid_coor[:, [1, 0]]

                range_valid_mask = no_point_mask.clone()
                range_valid_mask[no_point_mask] = valid_mask
                
                # store projected coordinates for later use
                range_cam_coors[i, cam_it][range_valid_mask] = valid_coor

                # remove masked points
                valid_mask = valid_mask & no_point_range_mask
                valid_points = range_points[i, no_point_mask, :][valid_mask, :]
                valid_coor = this_coor[valid_mask, :]
                valid_coor[:, [0, 1]] = valid_coor[:, [1, 0]]

                range_valid_mask = no_point_mask.clone()
                range_valid_mask[no_point_mask] = valid_mask
                
                
                valid_coor[:, 0] = valid_coor[:, 0] * img_features.shape[-2] / resize_dims[0]
                valid_coor[:, 1] = valid_coor[:, 1] * img_features.shape[-1] / resize_dims[1]
                valid_coor = valid_coor.type(torch.long)
                
                # project image features to range image coordinates
                cam_range_features[i, :, range_valid_mask] += img_features[i, cam_it][:, valid_coor[:, 0], valid_coor[:, 1]]
                
                # pos embed for image features
                pos_embed = self.position_encoder(valid_points)
                img_features[i, cam_it][:, valid_coor[:, 0], valid_coor[:, 1]] +=  pos_embed.permute(1, 0)
        
        return torch.cat((range_image, cam_range_features), dim = 1), img_features, range_cam_coors
        
