import torch.nn as nn
import torch
import torch.nn.functional as F
from mmdet.models import DETECTORS
from mmdet3d.models.builder import build_backbone, build_head, build_neck
from mmdet3d.models.utils.pos_embed import get_2d_sincos_pos_embed
from mmdet3d.losses.sup_con_loss import SupConLoss

import torch.distributed as dist
from collections import OrderedDict


import numpy as np
from PIL import Image
import math

import matplotlib.pyplot as plt
import os



@DETECTORS.register_module()
class PointTriplane(nn.Module):
    """Pretraining Point Triplane

    Args:
        point_triplane_projector (dict): Class that projects point features to triplane
        camera_encoder (dict): Camera encoder class
        triplane_encoder (dict): 2D network to encode triplane
        fpn (dict): Feature pyramid network to combine multiscale features of triplane encoder
        lidar_decoder (dict): Decoder for range image reconstruction
        camera_decoder (dict): Decoder for image reconstruction
        surface_decoder (dict): Decoder for surface reconstruction
        color_decoder (dict): Decoder for color reconstruction
        contrstive (bool): Whether to use multi-positive contrastive loss
        voxel_size (list): Voxel size
        pc_range (list): Pointcloud range
        checkpoint_path (str): path for initial weights

    """
    def __init__(self, 
                 point_triplane_projector,
                 camera_encoder,
                 triplane_encoder,
                 fpn,
                 lidar_decoder = None,
                 camera_decoder = None,
                 surface_decoder = None,
                 color_decoder = None,
                 contrastive = False,
                 voxel_size = None,
                 pc_range = None,
                 checkpoint_path = None,
                 train_cfg = None,
                 test_cfg = None
                 ):
        super(PointTriplane, self).__init__()
        self.point_triplane_projector = build_backbone(point_triplane_projector)
        self.camera_encoder = build_backbone(camera_encoder)
        self.triplane_encoder = build_backbone(triplane_encoder)
        self.fpn = build_neck(fpn)
        
        self.camera_decoder = None
        self.img_reconstruction = False
        self.lidar_decoder = None
        self.surface_decoder = None
        self.color_decoder = None
        self.contrastive = contrastive
        if contrastive:
            self.contrastive_loss = SupConLoss()

        if camera_decoder is not None:
            self.camera_decoder = build_head(camera_decoder)
            self.img_reconstruction = True
        
        if lidar_decoder is not None:
            self.lidar_decoder = build_head(lidar_decoder)
        
        if surface_decoder is not None:
            self.surface_decoder = build_head(surface_decoder)
        
        if color_decoder is not None:
            self.color_decoder = build_head(color_decoder)
            

        self.relu = nn.ReLU(inplace=True)

        self.voxel_size = voxel_size
        self.pc_range = pc_range
        self.ckpt_path = checkpoint_path
        self.count = 0
        
        if self.ckpt_path is not None:
            self.load_weights()
            print("Loaded weights from ", self.ckpt_path)
        
            
    
    def load_weights(self):
        """Initialize encoder with pretrained weights 

        """
        ckpt_state = torch.load(self.ckpt_path)['state_dict']
        self._load_state_dict(ckpt_state)

        
    def _load_state_dict(self, model_state_disk, strict=False):
        """Loads weights from dictionary that match keys and shape. 

        Args:
            model_state_disk (dict): State dictionary of model that we want to load
            
        Returns:
            state_dict (dict): Local cache of state dict
            update_model_state (dict): Updated model state

        """
        state_dict = self.state_dict()  # local cache of state_dict
        update_model_state = {}

        for key, val in model_state_disk.items():
            if key in state_dict and state_dict[key].shape == val.shape:
                update_model_state[key] = val
        
        if strict:
            self.load_state_dict(update_model_state)
        else:
            state_dict.update(update_model_state)
            self.load_state_dict(state_dict)
        return state_dict, update_model_state
    
    
    def voxelize_points(self, points):
        """Convert points into voxels. 

        Args:
            points (torch.tensor): Lidar points
            
        Returns:
            cropped_points (torch.tensor): Points that lie withing range
            grid_int (torch.tensor): Grid indexes for each point

        """
       
        cropped_points = []
        grid_ind = []
        for pts in points:
            crop_mask = (pts[..., 0] > self.pc_range[0]) & (pts[..., 0] < self.pc_range[3]) & \
                        (pts[..., 1] > self.pc_range[1]) & (pts[..., 1] < self.pc_range[4]) & \
                        (pts[..., 2] > self.pc_range[2]) & (pts[..., 2] < self.pc_range[5])
            
            cropped_pts = pts[crop_mask]
            voxel_ind = torch.zeros((cropped_pts.shape[0], 3), device = pts.device, dtype = pts.dtype)
            voxel_ind[..., 0] = (cropped_pts[..., 0] - self.pc_range[0]) / self.voxel_size[0]
            voxel_ind[..., 1] = (cropped_pts[..., 1] - self.pc_range[1]) / self.voxel_size[1]
            voxel_ind[..., 2] = (cropped_pts[..., 2] - self.pc_range[2]) / self.voxel_size[2]
            
            cropped_points.append(cropped_pts)
            grid_ind.append(voxel_ind.type(torch.int))
        
        return cropped_points, grid_ind
        
        
    def point_to_cam(self, points, img_features, img_metas):
        """Project points to pixel locations and sample features. 

        Args:
            points (torch.tensor): Lidar points
            img_features (torch.tensor): Image features extracted with encoder
            img_metas (dict): Meta informaton of images
            
        Returns:
            cam_point_features (torch.tensor): Image features for each point

        """
        resize_dims = img_metas[0]["img_shape"][::-1]
        lidar2imgs = []
        img_augs = []
        for img_meta in img_metas:
            lidar2imgs.append(img_meta['lidar2image'])
            img_augs.append(img_meta["imgs_aug"])

        lidar2imgs = np.asarray(lidar2imgs)
        lidar2imgs = points[0].new_tensor(lidar2imgs)
        cam_point_features = []
        
        for i, pts in enumerate(points):
            point_feature = torch.zeros((pts.shape[0], img_features.shape[2]), dtype = img_features.dtype, device = img_features.device)
            lidar2img = lidar2imgs[i]
            hom_points = torch.cat((pts[:, 0:3], torch.ones_like(pts[..., :1])), -1)
            cam_points = torch.einsum("cij, hj->chi", lidar2img, hom_points)
            cam_points = cam_points[..., 0:2] / torch.maximum(
                cam_points[..., 2:3], torch.ones_like(cam_points[..., 2:3]) * 1e-5)
            
            
            num_cam = lidar2imgs.shape[1]
            resize = [aug["resize"] for aug in img_augs[i]]
            crop = [aug["crop"] for aug in img_augs[i]]
            flip = [aug["flip"] for aug in img_augs[i]]
                
            for cam_it in range(num_cam):
                this_coor = cam_points[cam_it]
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

                # depth_coords = this_coor[:, :2].type(torch.long)
                valid_mask = ((this_coor[:, 1] < resize_dims[0])
                            & (this_coor[:, 0] < resize_dims[1])
                            & (this_coor[:, 1] >= 0)
                            & (this_coor[:, 0] >= 0))
                
                valid_coor = this_coor[valid_mask, :]
                valid_coor[:, [0, 1]] = valid_coor[:, [1, 0]]

                valid_coor[:, 0] = 2 * valid_coor[:, 0] / H - 1
                valid_coor[:, 1] = 2 * valid_coor[:, 1] / W - 1 
                
                features = F.grid_sample(img_features[i][cam_it][None], valid_coor[None, :, None]).squeeze(0).squeeze(-1)
                point_feature[valid_mask] += features.permute(1, 0).contiguous()
                # vis_proj(img[i][cam_it], this_coor[valid_mask], pts[valid_mask][:, 0:3], "/cluster/scratch/scharyyev/thesis/vis/", str(i) + "_" + str(cam_it) + ".png")

            cam_point_features.append(point_feature)
        
        return cam_point_features

    def cam_rec_feat(self, points, points_feat, img_metas):
        """Project triplane features to pixels for reconstruction. 

        Args:
            points (torch.tensor): Lidar points
            points_features (torch.tensor): Point features extracted from triplane
            img_metas (dict): Meta informaton of images
            
        Returns:
            cam_rec_features (torch.tensor): Projected triplane features in image plane

        """
        resize_dims = img_metas["img_shape"][::-1]
        
        lidar2img = img_metas['lidar2image']
        img_augs = img_metas["imgs_aug"]

        lidar2img = np.asarray(lidar2img)
        lidar2img = points.new_tensor(lidar2img)
        num_cam = lidar2img.shape[0]
        
        cam_rec_features = torch.zeros((num_cam, points_feat.shape[0], resize_dims[0], resize_dims[1]), device = points_feat.device)

        lidar2img = lidar2img
        hom_points = torch.cat((points, torch.ones_like(points[..., :1])), -1)
        cam_points = torch.einsum("cij, hj->chi", lidar2img, hom_points)
        cam_points = cam_points[..., 0:2] / torch.maximum(
            cam_points[..., 2:3], torch.ones_like(cam_points[..., 2:3]) * 1e-5)
        
        
        resize = [aug["resize"] for aug in img_augs]
        crop = [aug["crop"] for aug in img_augs]
        flip = [aug["flip"] for aug in img_augs]
            
        for cam_it in range(num_cam):
            this_coor = cam_points[cam_it]
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

            # depth_coords = this_coor[:, :2].type(torch.long)
            valid_mask = ((this_coor[:, 1] < resize_dims[0])
                        & (this_coor[:, 0] < resize_dims[1])
                        & (this_coor[:, 1] >= 0)
                        & (this_coor[:, 0] >= 0))
            
            valid_coor = this_coor[valid_mask, :].type(torch.long)
            valid_coor[:, [0, 1]] = valid_coor[:, [1, 0]]
            cam_rec_features[cam_it][:, valid_coor[:, 0], valid_coor[:, 1]] = points_feat[:, valid_mask]
        
        return cam_rec_features
        
    
        
    def forward(self,
                img,
                img_metas,
                points,
                return_loss=True, 
                pretrain=False,
                out_dir=None,
                show_pretrain = False,
                pos = None,
                **kwargs,
                ):
        """Forward call. 

        Args:
            img (torch.tensor): Images
            points (torch.tensor): Points
            img_metas (dict): Meta information of images
            return_loss (bool): Decides whether to train or test
            
        Returns:
            (dict): Loss

        """

        # train
        if return_loss:
            # voxelize points
            points, grid_ind = self.voxelize_points(points)
            
            # extract image features
            B, N, C, H, W = img.size()
            img = img.view(B * N, C, H, W)
            img_features, _ = self.camera_encoder(img)
            _, c, h, w = img_features.shape
            img_features = img_features.view(B, N, c, h, w)
            
            # project features to triplane
            cam_point_features = self.point_to_cam(points, img_features, img_metas)
            tpv = self.point_triplane_projector(points, grid_ind, cam_point_features)

            # encode triplae
            triplane = []
            for tp in tpv:
                tp_features = self.triplane_encoder(tp)
                triplane.append(self.fpn(tp_features))  

            # compute losses
            losses = {}
            
            # image reconstruction
            cam_features = []  
            if self.camera_decoder is not None:
                for i, pts in enumerate(points):
                    triplane_i = []
                    for tp in triplane:
                        triplane_i.append(tp[i][None])

                    coords = pts[:, 0:3]
                                 
                    features = self.sample_points_triplane(triplane_i, coords[None, None, ...]).squeeze()
                    
                    cam_features.append(self.cam_rec_feat(coords, features, img_metas[i]))
            
            
            cam_pred = torch.cat(cam_features, dim = 0)
            cam_pred = self.camera_decoder(cam_pred)
            camera_mask = torch.ones_like(img)

        
            losses['camera_loss'] = self.camera_decoder.forward_loss(img, cam_pred, camera_mask)
                
            
            # contrastive loss
            if self.contrastive:
                count = 0
                loss = 0
                for i, pts in enumerate(points):
                    triplane_i = []
                    for tp in triplane:
                        triplane_i.append(tp[i][None])
                    for cam in range(6):
                        # cam = torch.randint(6, (1, 1)).item()
                        coords = pts[:, 0:3]
                        index = 5 + cam
                        labels = pts[:, index]
                        valid_mask = labels > 0
                        coords = coords[valid_mask]
                        labels = labels[valid_mask].type(torch.int)
                        
                        if labels.shape[0] > 1:
                            features = self.sample_points_triplane(triplane_i, coords[None, None, ...]).squeeze()
                            features = features.permute(1, 0)
                            cl_loss = self.contrastive_loss(features, labels)
                            if cl_loss is not None:
                                loss += cl_loss
                                count += 1
                if count > 0:
                    losses['contrastive_loss'] = loss / count
                else:
                    losses['contrastive_loss'] = torch.tensor(0, dtype = torch.float32, device = img.device)

            # surface reconstruction       
            if self.surface_decoder is not None:
                point_features = []
                point_coords = []
                for i in range(len(points)):
                    triplane_i = []
                    for tp in triplane:
                        triplane_i.append(tp[i][None])
                    
                    coords = points[i][:, 0:3]
                    features = self.sample_points_triplane(triplane_i, coords[None, None, ...]).squeeze()
                    features = features.permute(1, 0)
                    point_features.append(features)
                    point_coords.append(coords)
                    
                    
                surface_targets = self.surface_decoder.create_targets(point_coords, point_features)
                surface_pred = self.surface_decoder(surface_targets)
                losses['surface_loss'] = surface_pred['surface_loss']
    
            
            return losses
        


    def sample_points_triplane(self, triplane, points):
        """Sample features for points in triplane. 

        Args:
            triplane (list[torch.tensor]): Triplane
            points (torch.tensor): Points
            
        Returns:
            sampled_feat (torch.tensor): Triplane feature for each point

        """
        voxel_coors = torch.zeros_like(points)
        voxel_coors[..., 0] = (points[..., 0] - self.pc_range[0]) / self.voxel_size[0]
        voxel_coors[..., 1] = (points[..., 1] - self.pc_range[1]) / self.voxel_size[1]
        voxel_coors[..., 2] = (points[..., 2] - self.pc_range[2]) / self.voxel_size[2]

        triplane_size = self.point_triplane_projector.grid_size
        voxel_coors[..., 0] = voxel_coors[..., 0] / (triplane_size[0] / 2) - 1
        voxel_coors[..., 1] = voxel_coors[..., 1] / (triplane_size[1] / 2) - 1
        voxel_coors[..., 2] = voxel_coors[..., 2] / (triplane_size[2] / 2) - 1

        xy_feat = F.grid_sample(triplane[0], voxel_coors[..., [0, 1]], mode='bilinear', padding_mode='zeros')
        yz_feat = F.grid_sample(triplane[1], voxel_coors[..., [1, 2]], mode='bilinear', padding_mode='zeros')
        xz_feat = F.grid_sample(triplane[2], voxel_coors[..., [0, 2]], mode='bilinear', padding_mode='zeros')

        sampled_feat = xy_feat + yz_feat + xz_feat
        
        return sampled_feat
    
        
    def train_step(self, data, optimizer):
        """The iteration step during training.

        This method defines an iteration step during training, except for the
        back propagation and optimizer updating, which are done in an optimizer
        hook. Note that in some complicated cases or models, the whole process
        including back propagation and optimizer updating is also defined in
        this method, such as GAN.

        Args:
            data (dict): The output of dataloader.
            optimizer (:obj:`torch.optim.Optimizer` | dict): The optimizer of
                runner is passed to ``train_step()``. This argument is unused
                and reserved.

        Returns:
            dict: It should contain at least 3 keys: ``loss``, ``log_vars``, \
                ``num_samples``.

                - ``loss`` is a tensor for back propagation, which can be a \
                weighted sum of multiple losses.
                - ``log_vars`` contains all the variables to be sent to the
                logger.
                - ``num_samples`` indicates the batch size (when the model is \
                DDP, it means the batch size on each GPU), which is used for \
                averaging the logs.
        """
        losses = self(**data)
        loss, log_vars = self._parse_losses(losses)

        outputs = dict(loss=loss, log_vars=log_vars, num_samples=len(data['img_metas']))

        return outputs
    
    def _parse_losses(self, losses):
        """Parse the raw outputs (losses) of the network.

        Args:
            losses (dict): Raw output of the network, which usually contain
                losses and other necessary infomation.

        Returns:
            tuple[Tensor, dict]: (loss, log_vars), loss is the loss tensor \
                which may be a weighted sum of all losses, log_vars contains \
                all the variables to be sent to the logger.
        """
        log_vars = OrderedDict()
        for loss_name, loss_value in losses.items():
            if isinstance(loss_value, torch.Tensor):
                log_vars[loss_name] = loss_value.mean()
            elif isinstance(loss_value, list):
                log_vars[loss_name] = sum(_loss.mean() for _loss in loss_value)
            else:
                raise TypeError(f"{loss_name} is not a tensor or list of tensors")

        loss = sum(_value for _key, _value in log_vars.items() if "loss" in _key)

        log_vars["loss"] = loss
        for loss_name, loss_value in log_vars.items():
            # reduce loss when distributed training
            if dist.is_available() and dist.is_initialized():
                loss_value = loss_value.data.clone()
                dist.all_reduce(loss_value.div_(dist.get_world_size()))
            log_vars[loss_name] = loss_value.item()

        return loss, log_vars


