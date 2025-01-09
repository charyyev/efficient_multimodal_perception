import torch.nn as nn
import torch
import torch.nn.functional as F
from mmdet.models import DETECTORS
from mmdet3d.models.builder import build_backbone, build_head, build_neck
from mmdet3d.models.utils.pos_embed import get_2d_sincos_pos_embed


import torch.distributed as dist
from collections import OrderedDict



import numpy as np
from PIL import Image
import math

import matplotlib.pyplot as plt
import os



@DETECTORS.register_module()
class TriplaneElev(nn.Module):
    """Elevation model using triplane representation

    Args:
        encoder (dict): Encoder for both range image and image
        neck (dict): Module that maps modality specific features to triplane
        decoder (dict): Decoder for elevation
        volume (list): Voxel dimensions
        voxel_size (list): Voxel size
        pc_range (list): Pointcloud range
        ckpt_path (str): Path for initial weights
        triplane_range (list): Range of triplane

    """
    def __init__(self, 
                 encoder,
                 neck,
                 decoder,
                 ckpt_path,
                 volume,
                 voxel_size = None,
                 pc_range = None,
                 triplane_range = None,
                 train_cfg = None,
                 test_cfg = None
                 ):
        super(TriplaneElev, self).__init__()
        self.encoder = build_backbone(encoder)
        self.neck = build_neck(neck)
        self.decoder = build_head(decoder)

        self.relu = nn.ReLU(inplace=True)

        self.voxel_size = voxel_size
        self.pc_range = pc_range
        self.volume = volume
        self.triplane_range = triplane_range
        self.test_num = 0
        self.ckpt_path = ckpt_path

        self.init_weights()
    
    
    def init_weights(self):
        """Initialize encoder with pretrained weights and freeze  

        """
        ckpt_state = torch.load(self.ckpt_path)['state_dict']
        self._load_state_dict(ckpt_state)

        self.encoder.eval()
        self.neck.eval()
        
        
        for param in self.encoder.parameters():
            param.requires_grad = False
        
        for param in self.neck.parameters():
            param.requires_grad = False


        

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

            
    
    def get_reference_points(self, volume, bs, device='cuda', dtype=torch.float):
        """Get the reference points used in SCA and TSA.
        Args:
            H, W, Z: spatial shape of volume.
            device (obj:`device`): The device where
                reference_points should be.
        Returns:
            Tensor: reference points used in decoder, has \
                shape (bs, num_keys, num_levels, 2).
        """
        H, W, Z = volume
        zs = torch.linspace(0.5, Z - 0.5, Z, dtype=dtype,
                            device=device).view(1, 1, Z).expand(H, W, Z)
        xs = torch.linspace(0.5, W - 0.5, W, dtype=dtype,
                            device=device).view(1, W, 1).expand(H, W, Z)
        ys = torch.linspace(0.5, H - 0.5, H, dtype=dtype,
                            device=device).view(H, 1, 1).expand(H, W, Z)
        ref_3d = torch.stack((ys, xs, zs), -1)
        
        ref_3d = ref_3d[None].repeat(bs, 1, 1, 1, 1)
        return ref_3d
    
    
    @torch.no_grad()
    def test_pretrain(self,
                      img,
                      range_image,
                      img_metas,
                      range_points,
                      elevation,
                      elevation_mask,
                      out_dir):
        """Test function. 

        Args:
            img (torch.tensor): Images
            range_image (torch.tensor): Range image
            range_points (torch.tensor): Points in the form of range image
            img_metas (dict): Meta information of images
            elevation (torch.tensor): Ground truth elevation
            elevation_mask (torch.tensor): Mask for sparse elevation
            out_dir (str): Save location for visualization
            
        Returns:
            (dict): Metrics

        """
        self.test_num += 1
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

        # combine features spatially
        B, N, C, H, W = image_features.shape
        image_features = image_features.permute(0, 2, 1, 3, 4).contiguous().view(B, C, -1, W)
        combined_features = torch.cat((range_features, image_features), dim = 2)
        
        # triplane encoding
        triplane = self.neck(combined_features)
        B, C, H, W = triplane.shape
        triplane = triplane.view(B, 3, -1, H, W)

        # sample feature for voxel centers
        points = self.get_reference_points(self.volume, B, triplane.device)
        points[..., 0] = points[..., 0] * self.voxel_size[0] + self.pc_range[0]
        points[..., 1] = points[..., 1] * self.voxel_size[1] + self.pc_range[1]
        points[..., 2] = points[..., 2] * self.voxel_size[2] + self.pc_range[2]

        voxel_feat = self.sample_points_triplane(triplane, points)
        
        # prediction
        pred = self.decoder(voxel_feat)
        mse = self.decoder.loss(pred, elevation.unsqueeze(1), elevation_mask.unsqueeze(1))
        
        # vis
        if self.test_num < 10:
            vis_triplane(triplane, out_dir, self.test_num)
            np.savez(os.path.join(out_dir, str(self.test_num)), pred_elevation = pred.detach().cpu().squeeze().numpy(),
                    gt_elevation = elevation[0].cpu().squeeze().numpy(), mask = elevation_mask[0].cpu().squeeze().numpy(), points = range_points.reshape(-1, 3).cpu().numpy())
        
        return [{
            "elevation_mse": mse["loss"]
        }]


        
    
    def forward(self,
                img,
                range_image,
                img_metas,
                range_points,
                elevation,
                elevation_mask,
                return_loss=True, 
                pretrain=False,
                out_dir=None,
                show_pretrain = False,
                **kwargs,
                ):
        """Forward call. 

        Args:
            img (torch.tensor): Images
            range_image (torch.tensor): Range image
            range_points (torch.tensor): Points in the form of range image
            img_metas (dict): Meta information of images
            elevation (torch.tensor): Ground truth elevation
            elevation_mask (torch.tensor): Mask for sparse elevation
            out_dir (str): Save location for visualization
            return_loss (bool): Determines whether to train or test
            
        Returns:
            loss (dict): Loss

        """
        # train
        if return_loss:
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

            # combine features spatially
            B, N, C, H, W = image_features.shape
            image_features = image_features.permute(0, 2, 1, 3, 4).contiguous().view(B, C, -1, W)
            combined_features = torch.cat((range_features, image_features), dim = 2)
            
            # encode triplane
            triplane = self.neck(combined_features)
            B, C, H, W = triplane.shape
            triplane = triplane.view(B, 3, -1, H, W)

            # Sample features for voxel centers from triplane
            points = self.get_reference_points(self.volume, B, triplane.device)
            points[..., 0] = points[..., 0] * self.voxel_size[0] + self.pc_range[0]
            points[..., 1] = points[..., 1] * self.voxel_size[1] + self.pc_range[1]
            points[..., 2] = points[..., 2] * self.voxel_size[2] + self.pc_range[2]

            voxel_feat = self.sample_points_triplane(triplane, points)

            # prediction
            pred = self.decoder(voxel_feat)
            loss = self.decoder.loss(pred, elevation.unsqueeze(1), elevation_mask.unsqueeze(1))
            
            return loss
            
        # test
        else:
            return self.test_pretrain(img, range_image, img_metas, range_points, elevation, elevation_mask, out_dir)


    def sample_points_triplane(self, triplane, points):
        """Sample features for points in triplane. 

        Args:
            triplane (torch.tensor): Triplane
            points (torch.tensor): Points
            
        Returns:
            sampled_feat (torch.tensor): Triplane feature for each point

        """
        voxel_coors = torch.zeros_like(points)
        voxel_coors[..., 0] = (points[..., 0] - self.triplane_range[0]) / self.voxel_size[0]
        voxel_coors[..., 1] = (points[..., 1] - self.triplane_range[1]) / self.voxel_size[1]
        voxel_coors[..., 2] = (points[..., 2] - self.triplane_range[2]) / self.voxel_size[2]

        voxel_coors = voxel_coors / (triplane.shape[-1] / 2) - 1
        b, h, w, d, p = voxel_coors.shape
        voxel_coors = voxel_coors.view(b, h, w*d, p)

        xy_feat = F.grid_sample(triplane[:, 0], voxel_coors[..., [0, 1]], mode='bilinear', padding_mode='zeros')
        yz_feat = F.grid_sample(triplane[:, 1], voxel_coors[..., [1, 2]], mode='bilinear', padding_mode='zeros')
        xz_feat = F.grid_sample(triplane[:, 2], voxel_coors[..., [0, 2]], mode='bilinear', padding_mode='zeros')

        sampled_feat = xy_feat + yz_feat + xz_feat
        sampled_feat = sampled_feat.view(b, -1, h, w, d)
        
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


def vis_triplane(triplane, out_dir, count):
    """Visualize triplane with PCA. 

    Args:
        triplane (torch.tensor): Triplane
        out_dir (str): Save directory
        count (int): Number to save

    """
    from sklearn.decomposition import PCA
    triplane = triplane.permute(0, 1, 3, 4, 2).detach().cpu().squeeze().numpy()

    plt.rcParams['figure.figsize'] = [24, 12]
    fig, axes = plt.subplots(
            nrows=3, ncols=1
        )
    for row in range(3):
        axes[row].axis("off")


    for i in range(3):
        # Reshape the BEV to 2D array (128*128, 32)
        bev_reshaped = triplane[i].reshape(-1, 32)

        # Apply PCA to reduce to 3 components
        pca = PCA(n_components=3)
        bev_pca = pca.fit_transform(bev_reshaped)

        # Reshape back to 2D spatial dimensions (128, 128, 3)
        bev_pca_img = bev_pca.reshape(128, 128, 3)

        # Normalize for visualization
        bev_pca_img = (bev_pca_img - bev_pca_img.min()) / (bev_pca_img.max() - bev_pca_img.min())

        if i == 0:
            title = "xy"
        elif i == 1:
            title = "yz"
        else:
            title = "xz"

        axes[i].imshow(bev_pca_img)
        axes[i].set_title(title)
    
    os.makedirs(os.path.join(out_dir, "triplane"), exist_ok=True)
    plt.savefig(f"{out_dir}/triplane/{count}.png")
    plt.close()
   

