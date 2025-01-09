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
class TriplaneMAE(nn.Module):
    """Pretraining main model

    Args:
        encoder (dict): Encoder for both range image and image
        neck (dict): Module that maps modality specific features to triplane
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
                 encoder,
                 neck = None,
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
        super(TriplaneMAE, self).__init__()
        self.encoder = build_backbone(encoder)
        self.neck = build_neck(neck)
        self.camera_decoder = None
        self.lidar_decoder = None
        self.surface_decoder = None
        self.color_decoder = None
        self.contrastive = contrastive
        if contrastive:
            self.contrastive_loss = SupConLoss()

        if camera_decoder is not None:
            self.camera_decoder = build_head(camera_decoder)
        
        if lidar_decoder is not None:
            self.lidar_decoder = build_head(lidar_decoder)
        
        if surface_decoder is not None:
            self.surface_decoder = build_head(surface_decoder)
        
        if color_decoder is not None:
            self.color_decoder = build_head(color_decoder)
            

        self.relu = nn.ReLU(inplace=True)
        self.proj_cam_downsample = nn.Conv2d(32, 768, kernel_size=1)
        self.proj_range_downsample = nn.Conv2d(32, 768, kernel_size=1)

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
    


    
    @torch.no_grad()
    def test_pretrain(self,
                      img, 
                      range_image,
                      range_points, 
                      img_metas,
                      show_pretrain,
                      out_dir):
        """Test function. 

        Args:
            img (torch.tensor): Images
            range_image (torch.tensor): Range image
            img_metas (dict): Meta information of images
            range_points (torch.tensor): Points in the form of range image
            out_dir (str): Save location for visualization
            
        Returns:
            list(dict): Losses

        """
        # extract features
        range_features, range_mask, image_features, img_mask, range_cam_coors = self.encoder(img, range_image, img_metas, range_points)
        orig_range_mask = torch.clone(range_mask)
        
        # add pos embed
        range_pos_embed = get_2d_sincos_pos_embed(768, range_features.shape[-2:])
        cam_pos_embed = get_2d_sincos_pos_embed(768, image_features.shape[-2:])
        range_pos_embed = torch.from_numpy(range_pos_embed).to(range_features.device)
        cam_pos_embed = torch.from_numpy(cam_pos_embed).to(range_features.device)
        range_pos_embed = range_pos_embed.permute(1, 0).view(768, range_features.shape[-2], -1)
        cam_pos_embed = cam_pos_embed.permute(1, 0).view(768, image_features.shape[-2], -1)

        range_features += range_pos_embed
        image_features += cam_pos_embed

        # combine features in spatial dimension
        B, N, C, H, W = image_features.shape
        image_features = image_features.permute(0, 2, 1, 3, 4).contiguous().view(B, C, -1, W)
        combined_features = torch.cat((range_features, image_features), dim = 2)
        
        # triplane encoding
        triplane = self.neck(combined_features)
        B, C, H, W = triplane.shape
        triplane = triplane.view(B, 3, -1, H, W)

        # adjust masks
        h_up_ratio = int(range_points.shape[1] / range_mask.shape[2])
        w_up_ratio = int(range_points.shape[2] / range_mask.shape[3])
        range_mask = range_mask.repeat_interleave(h_up_ratio, dim = 2).repeat_interleave(w_up_ratio, dim = 3)
        point_mask = range_mask & (range_image > 0)
        
        # image reconstruction
        range_proj_feat = self.sample_points_triplane(triplane, range_points)  
        B, N, C, H, W = img.shape  
        losses = {}
        if self.camera_decoder is not None:
            range_cam_coors = range_cam_coors.long()
            cam_proj_feat = torch.zeros(B, N, range_proj_feat.shape[1], H, W, device = img.device)
            for b in range(B):
                for cam_it in range(N):
                    cam_coors = range_cam_coors[b, cam_it]
                    cam_coors_valid = cam_coors[..., 0] > 0
                    proj_feat = range_proj_feat[b]
                    cam_coors = cam_coors[cam_coors_valid, :]
                    proj_feat = proj_feat[:, cam_coors_valid]

                    cam_proj_feat[b, cam_it][:, cam_coors[:, 0], cam_coors[:, 1]] = proj_feat
            

            cam_pred = cam_proj_feat.view(B * N, -1, H, W)
            cam_pred = self.camera_decoder(cam_pred)

            img = img.view(B * N, C, H, W)
            camera_mask = torch.ones_like(img)

        
            losses['camera_loss'] = self.camera_decoder.forward_loss(img, cam_pred, camera_mask)
            img_mask = img_mask.repeat_interleave(camera_mask.shape[2] // img_mask.shape[2], dim = 2)
            img_mask = img_mask.repeat_interleave(camera_mask.shape[3] // img_mask.shape[3], dim = 3)
            camera_mask = camera_mask.type(torch.bool) & img_mask

            if show_pretrain and self.count < 20:
                vis_image(img, cam_pred, camera_mask.int(), self.camera_decoder, out_dir, self.count)
            
            img = img.view(B, N, C, H, W)
        
        # color reconstruction
        if self.color_decoder is not None:
            points = []
            latents = []
            colors = []

            range_cam_coors = range_cam_coors.long()
            for b in range(B):
                for cam_it in range(N):
                    cam_coors = range_cam_coors[b, cam_it]
                    cam_coors_valid = cam_coors[..., 0] > 0
                    proj_feat = range_proj_feat[b]
                    pts = range_points[b][cam_coors_valid]
                    image = img[b, cam_it]
                    cam_coors = cam_coors[cam_coors_valid, :]
                    proj_feat = proj_feat[:, cam_coors_valid].permute(1, 0)
                    color = image[:, cam_coors[:, 0], cam_coors[:, 1]].permute(1, 0)

                    latents.append(proj_feat)
                    points.append(pts)
                    colors.append(color)
            
            
            points = torch.cat(points, dim = 0)
            latents = torch.cat(latents, dim = 0)
            colors = torch.cat(colors, dim = 0)
            pred_colors = self.color_decoder(latents)
            losses["color"] = F.mse_loss(pred_colors, colors)

            
        
        # surface reconstruction
        if self.surface_decoder is not None:
            if show_pretrain and self.count < 20:
                point_mask = (range_points==0).sum(3) != 3
                range_feat = range_proj_feat.permute(0, 2, 3, 1)

                points = range_points[point_mask]
                features = range_feat[point_mask]

                points_per_point = 20

                direction = 2 * torch.rand((points.shape[0], points_per_point, 3), device = points.device) - 1
                direction = F.normalize(direction, dim = 2)
                dist = torch.rand((points.shape[0], points_per_point, 1), device = points.device)
                points = points.unsqueeze(1).repeat(1, points_per_point, 1)
                features = features.unsqueeze(1).repeat(1, points_per_point, 1)
                offset = dist * direction
                target_points = points + offset

                features = features.reshape(-1, features.shape[-1])
                target_points = target_points.reshape(-1, 3)
                offset = offset.reshape(-1, 3)
                features = torch.cat((features, offset), dim = 1)

                surface = torch.sigmoid(self.surface_decoder.test_forward(features))
                occupied = surface > 0.5
                occupied_points = target_points[occupied]
                os.makedirs(os.path.join(out_dir, "surface"), exist_ok=True)
                np.save(f"{out_dir}/surface/{self.count}.bin", occupied_points.cpu().numpy())

        
        # range image reconstruction
        if self.lidar_decoder is not None:
            range_proj_feat *= point_mask.repeat(1, range_proj_feat.shape[1], 1, 1)
            range_pred = range_proj_feat
            # range_pred = range_pred.flatten(2).permute(0, 2, 1)
            range_pred = self.lidar_decoder(range_pred)
            
            eps = 0.5
            range_mask = ((range_points[..., 0] > self.pc_range[0] + eps) & (range_points[..., 1] > self.pc_range[1] + eps) & (range_points[..., 2] > self.pc_range[2] + eps) \
            & (range_points[..., 0] < self.pc_range[3] - eps) & (range_points[..., 1] < self.pc_range[4] - eps) & (range_points[..., 2] < self.pc_range[5] - eps))
            range_mask = range_mask.unsqueeze(1)

            losses['range_loss'] = self.lidar_decoder.forward_loss(range_image, range_pred, range_mask)
            orig_range_mask = (orig_range_mask).repeat_interleave(range_mask.shape[2] // orig_range_mask.shape[2], dim = 2)
            orig_range_mask = orig_range_mask.repeat_interleave(range_mask.shape[3] // orig_range_mask.shape[3], dim = 3)
            range_mask = range_mask & orig_range_mask

            if show_pretrain and self.count < 20:
                range_pred, range_mask = vis_range_image(range_image, range_pred, range_mask.int(), self.lidar_decoder, out_dir, self.count)
                save_points(range_points, range_pred, range_mask, out_dir, self.count)

        
        self.count += 1
        
        return [losses]
    
        
    def forward(self,
                img,
                range_image,
                img_metas,
                range_points,
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
            range_image (torch.tensor): Range image
            img_metas (dict): Meta information of images
            range_points (torch.tensor): Points in the form of range image
            points (torch.tensor): Lidar points
            return_loss (bool): Decides whether to train or test
            out_dir (str): Save location for visualization
            
        Returns:
            loss (dict): Losses

        """

        # train
        if return_loss:
            # crop points outside range
            crop_mask = (range_points[..., 0] > self.pc_range[0]) & (range_points[..., 0] < self.pc_range[3]) & \
                        (range_points[..., 1] > self.pc_range[1]) & (range_points[..., 1] < self.pc_range[4]) & \
                        (range_points[..., 2] > self.pc_range[2]) & (range_points[..., 2] < self.pc_range[5])

            
            range_image = range_image * crop_mask.unsqueeze(1)
            range_points = range_points * crop_mask.unsqueeze(-1)
            
            # extract features
            range_features, range_mask, image_features, img_mask, range_cam_coors = self.encoder(img, range_image, img_metas, range_points)
            orig_range_mask = torch.clone(range_mask)
            
            # add pos embed
            range_pos_embed = get_2d_sincos_pos_embed(768, range_features.shape[-2:])
            cam_pos_embed = get_2d_sincos_pos_embed(768, image_features.shape[-2:])
            range_pos_embed = torch.from_numpy(range_pos_embed).to(range_features.device)
            cam_pos_embed = torch.from_numpy(cam_pos_embed).to(range_features.device)
            range_pos_embed = range_pos_embed.permute(1, 0).view(768, range_features.shape[-2], -1)
            cam_pos_embed = cam_pos_embed.permute(1, 0).view(768, image_features.shape[-2], -1)

            range_features += range_pos_embed
            image_features += cam_pos_embed

            # combine features in spatial dimension
            B, N, C, H, W = image_features.shape
            image_features = image_features.permute(0, 2, 1, 3, 4).contiguous().view(B, C, -1, W)
            combined_features = torch.cat((range_features, image_features), dim = 2)
            
            # triplane encoding
            triplane = self.neck(combined_features)
            B, C, H, W = triplane.shape
            triplane = triplane.view(B, 3, -1, H, W)

            h_up_ratio = int(range_points.shape[1] / range_mask.shape[2])
            w_up_ratio = int(range_points.shape[2] / range_mask.shape[3])
            range_mask = range_mask.repeat_interleave(h_up_ratio, dim = 2).repeat_interleave(w_up_ratio, dim = 3)
            point_mask = range_mask & (range_image > 0)
            
            # image reconstruction
            range_proj_feat = self.sample_points_triplane(triplane, range_points)
            B, N, C, H, W = img.shape  
            losses = {}
            if self.camera_decoder is not None:
                range_cam_coors = range_cam_coors.long()
                cam_proj_feat = torch.zeros(B, N, range_proj_feat.shape[1], H, W, device = img.device)
                for b in range(B):
                    for cam_it in range(N):
                        cam_coors = range_cam_coors[b, cam_it]
                        cam_coors_valid = cam_coors[..., 0] > 0
                        proj_feat = range_proj_feat[b]
                        cam_coors = cam_coors[cam_coors_valid, :]
                        proj_feat = proj_feat[:, cam_coors_valid]

                        cam_proj_feat[b, cam_it][:, cam_coors[:, 0], cam_coors[:, 1]] = proj_feat
                

                cam_pred = cam_proj_feat.view(B * N, -1, H, W)
                cam_pred = self.camera_decoder(cam_pred)

                img = img.view(B * N, C, H, W)
                camera_mask = torch.ones_like(img)

            
                losses['camera_loss'] = self.camera_decoder.forward_loss(img, cam_pred, camera_mask)
                img = img.view(B, N, C, H, W)
            
            # color reconstruction
            if self.color_decoder is not None:
                points = []
                latents = []
                colors = []

                range_cam_coors = range_cam_coors.long()
                for b in range(B):
                    for cam_it in range(N):
                        cam_coors = range_cam_coors[b, cam_it]
                        cam_coors_valid = cam_coors[..., 0] > 0
                        proj_feat = range_proj_feat[b]
                        pts = range_points[b][cam_coors_valid]
                        image = img[b, cam_it]
                        cam_coors = cam_coors[cam_coors_valid, :]
                        proj_feat = proj_feat[:, cam_coors_valid].permute(1, 0)
                        color = image[:, cam_coors[:, 0], cam_coors[:, 1]].permute(1, 0)

                        latents.append(proj_feat)
                        points.append(pts)
                        colors.append(color)
                
                
                points = torch.cat(points, dim = 0)
                latents = torch.cat(latents, dim = 0)
                colors = torch.cat(colors, dim = 0)
                pred_colors = self.color_decoder(latents)
                losses["color"] = F.mse_loss(pred_colors, colors)

                # vis_point_color(points, colors)

            # contrastive loss
            if self.contrastive:
                count = 0
                loss = 0
                for i, pts in enumerate(points):
                    crop_mask = (pts[..., 0] > self.pc_range[0]) & (pts[..., 0] < self.pc_range[3]) & \
                                (pts[..., 1] > self.pc_range[1]) & (pts[..., 1] < self.pc_range[4]) & \
                                (pts[..., 2] > self.pc_range[2]) & (pts[..., 2] < self.pc_range[5])
                    pts = pts[crop_mask]
                    for cam in range(6):
                        # cam = torch.randint(6, (1, 1)).item()
                        coords = pts[:, 0:3]
                        index = 5 + cam
                        labels = pts[:, index]
                        valid_mask = labels > 0
                        coords = coords[valid_mask]
                        labels = labels[valid_mask].type(torch.int)
                        
                        if labels.shape[0] > 1:
                            features = self.sample_points_triplane(triplane[i][None, ...], coords[None, None, ...]).squeeze()
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
                surface_targets = self.surface_decoder.create_targets(range_points, range_proj_feat)
                surface_pred = self.surface_decoder(surface_targets)
                losses['surface_loss'] = surface_pred['surface_loss']

            # range image reconstruction
            if self.lidar_decoder is not None:
                range_proj_feat *= point_mask.repeat(1, range_proj_feat.shape[1], 1, 1)
                range_pred = range_proj_feat
                range_pred = self.lidar_decoder(range_pred)
                

                range_mask = torch.ones_like(range_image)

                losses['range_loss'] = self.lidar_decoder.forward_loss(range_image, range_pred, range_mask)
                
            
            return losses
        
        else:
            return self.test_pretrain(img, range_image, range_points, img_metas, show_pretrain, out_dir)


    def sample_points_triplane(self, triplane, points):
        """Sample features for points in triplane. 

        Args:
            triplane (torch.tensor): Triplane
            points (torch.tensor): Points
            
        Returns:
            sampled_feat (torch.tensor): Triplane feature for each point

        """
        voxel_coors = torch.zeros_like(points)
        voxel_coors[..., 0] = (points[..., 0] - self.pc_range[0]) / self.voxel_size[0]
        voxel_coors[..., 1] = (points[..., 1] - self.pc_range[1]) / self.voxel_size[1]
        voxel_coors[..., 2] = (points[..., 2] - self.pc_range[2]) / self.voxel_size[2]

        voxel_coors = voxel_coors / (triplane.shape[-1] / 2) - 1

        xy_feat = F.grid_sample(triplane[:, 0], voxel_coors[..., [0, 1]], mode='bilinear', padding_mode='zeros')
        yz_feat = F.grid_sample(triplane[:, 1], voxel_coors[..., [1, 2]], mode='bilinear', padding_mode='zeros')
        xz_feat = F.grid_sample(triplane[:, 2], voxel_coors[..., [0, 2]], mode='bilinear', padding_mode='zeros')

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

def show_image(image, title=''):
    imagenet_mean = np.array([0.485, 0.456, 0.406])
    imagenet_std = np.array([0.229, 0.224, 0.225])
    
    # image is [H, W, 3]
    assert image.shape[2] == 3
    plt.imshow(torch.clip((image * imagenet_std + imagenet_mean) * 255, 0, 255).int())
    plt.title(title, fontsize=16)
    plt.axis('off')
    return

def save_image(image, path, name):
    imagenet_mean = np.array([0.485, 0.456, 0.406])
    imagenet_std = np.array([0.229, 0.224, 0.225])
    
    image = (image * imagenet_std + imagenet_mean) * 255
    
    im = Image.fromarray(image.numpy().astype(np.uint8))
    im.save(os.path.join(path, name))

def save_range_image(image, path, name):
    plt.imsave(os.path.join(path, name), image.numpy())

    

def show_range_image(image, title=''):
    im = plt.imshow(image)
    plt.title(title, fontsize=16)
    plt.colorbar(im, ax=plt)

    # plt.axis('off')
    return

def vis_image(ori_img, pred_img, mask, decoder, out_dir, count):
    # ori_img = decoder.patchify(ori_img)
    # mean = ori_img.mean(dim=1, keepdim=True)
    # var = ori_img.var(dim=1, keepdim=True)
    # ori_img = decoder.unpatchify(ori_img)
    x = torch.einsum('nchw->nhwc', ori_img).detach().cpu()

    # run MAE
    # pred_img = pred_img * (var + 1.e-6)**.5 + mean
    y = decoder.unpatchify(pred_img)
    y = torch.einsum('nchw->nhwc', y).detach().cpu()

    # visualize the mask
    mask = (1-mask).detach()
    # mask = mask.unsqueeze(-1).repeat(1, 1, decoder.final_patch_size[0] * decoder.final_patch_size[1] * decoder.in_chans)  # (N, H*W, p*p*3)
    # mask = decoder.unpatchify(mask)  # 1 is removing, 0 is keeping
    mask = torch.einsum('nchw->nhwc', mask).detach().cpu()

    # masked image
    im_masked = x * (1 - mask)

    # MAE reconstruction pasted with visible patches
    im_paste = x * (1 - mask) + y * mask

    # make the plt figure larger
    plt.rcParams['figure.figsize'] = [24, 12]

    os.makedirs(os.path.join(out_dir, "raw"), exist_ok=True)
    
    for i in range(x.shape[0]):
        # save overlaid image
        imagenet_mean = np.array([0.485, 0.456, 0.406])
        imagenet_std = np.array([0.229, 0.224, 0.225])

        orig_image = (x[i] * imagenet_std + imagenet_mean)
        # orig_image = orig_image + mask[i] * 0.5
        orig_image[mask[i].bool()] = orig_image[mask[i].bool()] * 0.5 + 0.5
        orig_image = orig_image * 255
        im = Image.fromarray(orig_image.numpy().astype(np.uint8))
        im.save(os.path.join(os.path.join(out_dir, "raw"), f"scene{count}_cam{str(i)}_overlay.png"))

        plt.subplot(6, 4, i*4+1)
        show_image(x[i], "original")
        save_image(x[i], os.path.join(out_dir, "raw"), f"scene{count}_cam{str(i)}_original.png")

        plt.subplot(6, 4, i*4+2)
        show_image(im_masked[i], "masked")
        save_image(im_masked[i], os.path.join(out_dir, "raw"), f"scene{count}_cam{str(i)}_masked.png")

        plt.subplot(6, 4, i*4+3)
        show_image(y[i], "reconstruction")
        save_image(y[i], os.path.join(out_dir, "raw"), f"scene{count}_cam{str(i)}_reconstruction.png")

        plt.subplot(6, 4, i*4+4)
        show_image(im_paste[i], "reconstruction + visible")

    os.makedirs(os.path.join(out_dir, "cam"), exist_ok=True)
    plt.savefig(f"{out_dir}/cam/{count}.png")
    plt.close()

def vis_range_image(ori_img, pred_img, mask, decoder, out_dir, count):
    # ori_img = decoder.patchify(ori_img)
    # mean = ori_img.mean(dim=1, keepdim=True)
    # var = ori_img.var(dim=1, keepdim=True)
    # ori_img = decoder.unpatchify(ori_img)
    x = torch.einsum('nchw->nhwc', ori_img).detach().cpu()

    # pred_img = pred_img * (var + 1.e-6)**.5 + mean
    y = decoder.unpatchify(pred_img)
    y = torch.einsum('nchw->nhwc', y).detach().cpu()

    # visualize the mask
    mask = (1-mask).detach()
    # mask = mask.unsqueeze(-1).repeat(1, 1, decoder.final_patch_size[0] * decoder.final_patch_size[1] * decoder.in_chans)  # (N, H*W, p*p*3)
    # mask = decoder.unpatchify(mask)  # 1 is removing, 0 is keeping
    mask = torch.einsum('nchw->nhwc', mask).detach().cpu()

    # masked image
    im_masked = x * (1 - mask)

    # MAE reconstruction pasted with visible patches
    im_paste = x * (1 - mask) + y * mask

    # make the plt figure larger
    plt.rcParams['figure.figsize'] = [24, 12]
    fig, axes = plt.subplots(
            nrows=4, ncols=1
        )
    for row in range(4):
        axes[row].axis("off")

    
    x_max = torch.max(x)
    y_max = torch.max(y)
    x_min = torch.min(x)
    y_min = torch.min(y)
    # rmin = min(x_min, y_min)
    rmin = 0
    # rmax = max(x_max, y_max)
    rmax = 50
    im = axes[0].imshow(x[0], vmin = rmin, vmax = rmax)
    axes[0].set_title("original")
    # save_range_image(x[0], os.path.join(out_dir, "raw"), "original.png")
    # plt.colorbar(im)

    im1 = axes[1].imshow(im_masked[0], vmin = rmin, vmax = rmax)
    axes[1].set_title("masked")
    # save_range_image(im_masked[0], os.path.join(out_dir, "raw"), "masked.png")
    # plt.colorbar(im1, ax=axes[1])

    im2 = axes[2].imshow(y[0], vmin = rmin, vmax = rmax)
    axes[2].set_title("reconstruction")
    # save_range_image(y[0], os.path.join(out_dir, "raw"), "reconstruction.png")
    # plt.colorbar(im2, ax=axes[2])

    im3 = axes[3].imshow(im_paste[0], vmin = rmin, vmax = rmax)
    axes[3].set_title("reconstruction + visible")
    # plt.colorbar(im3, ax=axes[3])

    os.makedirs(os.path.join(out_dir, "range"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "range_cut"), exist_ok=True)
    plt.savefig(f"{out_dir}/range/{count}.png")
    plt.close()
    
    
    for i in range(8):
        fig, axes = plt.subplots(
            nrows=4, ncols=1
        )
        for row in range(4):
            axes[row].axis("off")

        rmin = 0
        # rmax = max(x_max, y_max)
        rmax = 50
        width = x[0].shape[1] // 8

        im = axes[0].imshow(x[0][:, i * width:(i + 1) * width, :], vmin = rmin, vmax = rmax)
        axes[0].set_title("original")


        im1 = axes[1].imshow(im_masked[0][:, i * width:(i + 1) * width, :], vmin = rmin, vmax = rmax)
        axes[1].set_title("masked")


        im2 = axes[2].imshow(y[0][:, i * width:(i + 1) * width, :], vmin = rmin, vmax = rmax)
        axes[2].set_title("reconstruction")


        im3 = axes[3].imshow(im_paste[0][:, i * width:(i + 1) * width, :], vmin = rmin, vmax = rmax)
        axes[3].set_title("reconstruction + visible")

        plt.savefig(f"{out_dir}/range_cut/image{count}_{i}.png")
        plt.close()

    return y, mask

def save_points(range_points, range_pred, mask,  out_dir, count):
    proj_fov_up = 10
    proj_fov_down = -30
    proj_W = 1024
    proj_H = 32

    # laser parameters
    fov_up = proj_fov_up / 180.0 * np.pi      # field of view up in rad
    fov_down = proj_fov_down / 180.0 * np.pi  # field of view down in rad
    fov = abs(fov_down) + abs(fov_up)  # get field of view total in rad

    range_points = range_points.detach().squeeze().cpu().numpy()
    range_pred = range_pred.detach().squeeze().cpu().numpy()
    mask = mask.detach().squeeze().cpu().numpy()

    i, j = np.meshgrid(np.arange(range_pred.shape[0]), np.arange(range_pred.shape[1]), indexing='ij')
    pred_points = np.stack((i, j, range_pred[i, j]), axis = 2)

    pred_points[..., 0] = pred_points[..., 0] / proj_H
    pred_points[..., 1] = pred_points[..., 1] / proj_W

    pred_points[..., 0] = (1 - pred_points[..., 0]) * fov - abs(fov_down)
    pred_points[..., 1] = (2 * pred_points[..., 1] - 1) * np.pi

    x = pred_points[..., 2] * np.sin(pred_points[..., 1]) * np.cos(pred_points[..., 0])
    y = pred_points[..., 2] * np.cos(pred_points[..., 1]) * np.cos(pred_points[..., 0])
    z = np.sin(pred_points[..., 0]) * pred_points[..., 2]

    pred_points = np.stack((x, y, z, 1 - mask), axis = 2)[range_pred > 0.001].reshape(-1, 4)
    gt_points = np.concatenate((range_points, 1 - np.expand_dims(mask, 2)), axis = 2).reshape(-1, 4)


    os.makedirs(os.path.join(out_dir, "points"), exist_ok=True)
    np.save(f"{out_dir}/points/pred_points_{count}.bin", pred_points)
    np.save(f"{out_dir}/points/gt_points_{count}.bin", gt_points)


def calculate_image_rmse(ori_img, pred_img, decoder):
    ori_img = decoder.patchify(ori_img)
    mean = ori_img.mean(dim=-1, keepdim=True)
    var = ori_img.var(dim=-1, keepdim=True)
    ori_img = decoder.unpatchify(ori_img)
    x = torch.einsum('nchw->nhwc', ori_img).detach()

    # run MAE
    pred_img = pred_img * (var + 1.e-6)**.5 + mean
    y = decoder.unpatchify(pred_img).detach()
    y = torch.einsum('nchw->nhwc', y)
    
    imagenet_mean = torch.tensor([0.485, 0.456, 0.406], device= y.device)
    imagenet_std = torch.tensor([0.229, 0.224, 0.225], device = y.device)

    x = x * imagenet_std + imagenet_mean
    y = y * imagenet_std + imagenet_mean

    x = x * 255
    y = y * 255

    # RMSE per image
    rmse = torch.mean((x - y) ** 2, dim = (1, 2, 3)).sqrt()
    rmse = rmse.mean()

    return rmse

def calculate_range_rmse(ori_img, pred_img, decoder):
    ori_img = decoder.patchify(ori_img)
    mean = ori_img.mean(dim=-1, keepdim=True)
    var = ori_img.var(dim=-1, keepdim=True)
    ori_img = decoder.unpatchify(ori_img)
    x = torch.einsum('nchw->nhwc', ori_img).detach().cpu()

    # run MAE
    pred_img = pred_img * (var + 1.e-6)**.5 + mean
    y = decoder.unpatchify(pred_img)
    y = torch.einsum('nchw->nhwc', y).detach().cpu()

    # RMSE per image
    rmse = torch.mean((x - y) ** 2).sqrt()

    return rmse

def vis_point_color(points, color):
    imagenet_mean = torch.tensor([0.485, 0.456, 0.406], device= points.device)
    imagenet_std = torch.tensor([0.229, 0.224, 0.225], device = points.device)

    colors = colors * imagenet_std + imagenet_mean
    
    color_point = torch.cat((points, colors), dim = 1)
    # np.save("/cluster/scratch/scharyyev/thesis/color_points/" + str(self.count) + ".bin", color_point.cpu().numpy())
    


def vis_proj(image, points, color, path, name):
    image = image.cpu().permute(1, 2, 0).numpy()
    points = points.cpu().numpy()
    color = color.cpu().numpy() 
    color = np.linalg.norm(color, axis = 1)
    color = color / color.max()
    os.makedirs(path, exist_ok=True)

    imagenet_mean = np.array([0.485, 0.456, 0.406])
    imagenet_std = np.array([0.229, 0.224, 0.225])
    
    image = (image * imagenet_std + imagenet_mean) * 255

    fig, ax = plt.subplots(1, 1, figsize=(9, 16))
    
    ax.imshow(image.astype(int))
    ax.scatter(points[:, 0], points[:, 1], c=color, s=5)
    ax.axis('off')

    plt.savefig(os.path.join(path, name))
