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
class PointTriplaneOcc(nn.Module):
    """Occupancy prediction with Point Triplane

    Args:
        point_triplane_projector (dict): Class that projects point features to triplane
        camera_encoder (dict): Camera encoder class
        triplane_encoder (dict): 2D network to encode triplane
        fpn (dict): Feature pyramid network to combine multiscale features of triplane encoder
        decoder (dict): Decoder for occupancy prediction
        ckpt_path (str): Checkpoint path for pretrained weights
        voxel_size (list): Voxel size
        occ_range (list): Range of points to consider for occupancy
        triplane_range (list): Range of points that triplane encodes
        triplane_voxel_size (list): Pixel size of triplane
        class_names (list): Class names for occupancy, used for evaluation
        freeze_encoder (bool): Whether to freeze triplane representation


    """
    def __init__(self, 
                 point_triplane_projector,
                 camera_encoder,
                 triplane_encoder,
                 fpn,
                 decoder,
                 ckpt_path,
                 voxel_size,
                 occ_range,
                 triplane_range,
                 triplane_voxel_size,
                 class_names = None,
                 freeze_encoder = True,
                 train_cfg = None,
                 test_cfg = None
                 ):
        super(PointTriplaneOcc, self).__init__()
        self.point_triplane_projector = build_backbone(point_triplane_projector)
        self.camera_encoder = build_backbone(camera_encoder)
        self.triplane_encoder = build_backbone(triplane_encoder)
        self.fpn = build_neck(fpn)
        self.decoder = build_head(decoder)
            
        self.relu = nn.ReLU(inplace=True)

        self.voxel_size = voxel_size
        self.occ_range = occ_range
        self.triplane_range = triplane_range
        self.triplane_voxel_size = triplane_voxel_size
        self.test_num = 0
        self.ckpt_path = ckpt_path
        self.class_names = class_names
        self.freeze_encoder = freeze_encoder

        self.init_weights()
        self.occ_bounds, self.ref_3d = self.roi()
        
            
    
    def init_weights(self):
        """Initialize encoder with pretrained weights, freeze if necessary  

        """
        ckpt_state = torch.load(self.ckpt_path)['state_dict']
        self._load_state_dict(ckpt_state)

        if self.freeze_encoder:
            self.point_triplane_projector.eval()
            self.camera_encoder.eval()
            self.triplane_encoder.eval()
            self.fpn.eval()
        
            for param in self.point_triplane_projector.parameters():
                param.requires_grad = False
            
            for param in self.camera_encoder.parameters():
                param.requires_grad = False
            
            for param in self.triplane_encoder.parameters():
                param.requires_grad = False
            
            for param in self.fpn.parameters():
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
            crop_mask = (pts[..., 0] > self.triplane_range[0]) & (pts[..., 0] < self.triplane_range[3]) & \
                        (pts[..., 1] > self.triplane_range[1]) & (pts[..., 1] < self.triplane_range[4]) & \
                        (pts[..., 2] > self.triplane_range[2]) & (pts[..., 2] < self.triplane_range[5])
            
            cropped_pts = pts[crop_mask]
            voxel_ind = torch.zeros((cropped_pts.shape[0], 3), device = pts.device, dtype = pts.dtype)
            voxel_ind[..., 0] = (cropped_pts[..., 0] - self.triplane_range[0]) / self.triplane_voxel_size[0]
            voxel_ind[..., 1] = (cropped_pts[..., 1] - self.triplane_range[1]) / self.triplane_voxel_size[1]
            voxel_ind[..., 2] = (cropped_pts[..., 2] - self.triplane_range[2]) / self.triplane_voxel_size[2]
            
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

                # point that lie in the image
                valid_mask = ((this_coor[:, 1] < resize_dims[0])
                            & (this_coor[:, 0] < resize_dims[1])
                            & (this_coor[:, 1] >= 0)
                            & (this_coor[:, 0] >= 0))
                
                valid_coor = this_coor[valid_mask, :]
                valid_coor[:, [0, 1]] = valid_coor[:, [1, 0]]

                valid_coor[:, 0] = 2 * valid_coor[:, 0] / H - 1
                valid_coor[:, 1] = 2 * valid_coor[:, 1] / W - 1 
                
                # sample feature with bilinear interpolation
                features = F.grid_sample(img_features[i][cam_it][None], valid_coor[None, :, None]).squeeze(0).squeeze(-1)
                point_feature[valid_mask] += features.permute(1, 0).contiguous()

            cam_point_features.append(point_feature)
        
        return cam_point_features

    
    @torch.no_grad()
    def test_pretrain(self,
                      img, 
                      points, 
                      img_metas,
                      occupancy,
                      out_dir):
        """Test function. 

        Args:
            img (torch.tensor): Images
            points (torch.tensor): Points
            img_metas (dict): Meta information of images
            occupancy (torch.tensor): Ground truth occupancy
            out_dir (str): Save location for visualization
            
        Returns:
            (dict): Metrics

        """

        # voxel centers to sample features from for decoder
        ref_3d = self.ref_3d.to(img.device).unsqueeze(0).repeat(len(points), 1, 1, 1, 1)
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

        # encode triplane
        triplane = []
        for tp in tpv:
            tp_features = self.triplane_encoder(tp)
            triplane.append(self.fpn(tp_features))  

        # sample features of voxel centers from triplane
        voxel_feat = self.sample_points_triplane(triplane, ref_3d)

        # Cut occupancy outside the range
        occ = occupancy[:, self.occ_bounds[0]: self.occ_bounds[2] + 1,self.occ_bounds[1]: self.occ_bounds[3] + 1]

        # prediction
        pred = self.decoder(voxel_feat)
        loss = self.decoder.loss(pred, occ)
        

        pred = torch.softmax(pred, dim = 1)
        pred = pred.argmax(dim = 1)

        # evaluation and visualization
        ious = evaluation_semantic(pred, occ, len(self.class_names) + 1)
        self.test_num += 1
        if self.test_num < 100:
            vis_triplane(triplane, out_dir, self.test_num)
            np.savez(os.path.join(out_dir, str(self.test_num)), pred_occ = pred.detach().cpu().squeeze().numpy(),
                    gt_occ = occ.cpu().squeeze().numpy(), points = points[0].cpu().numpy())
        
        return [{
            "CE": loss["loss"],
            "ious": ious
        }]
        
    def forward(self,
                img,
                img_metas,
                points,
                occupancy,
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
            occupancy (torch.tensor): Ground truth occupancy
            out_dir (str): Save location for visualization
            return_loss (bool): Decides whether to train or test
            
        Returns:
            (dict): Loss

        """
        # train
        if return_loss:
            # voxel centers to sample features from for decoder
            ref_3d = self.ref_3d.to(img.device).unsqueeze(0).repeat(len(points), 1, 1, 1, 1)
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

            # sample features of triplane from voxel centers
            voxel_feat = self.sample_points_triplane(triplane, ref_3d)
            
            # remove occupancy outside range
            occ = occupancy[:, self.occ_bounds[0]: self.occ_bounds[2] + 1,self.occ_bounds[1]: self.occ_bounds[3] + 1]

            # prediction
            pred = self.decoder(voxel_feat)
            loss = self.decoder.loss(pred, occ)
            
            return loss
            
        # test
        else:
            return self.test_pretrain(img, points, img_metas, occupancy, out_dir)


    def roi(self):
        """Computes occupancy region of interest and voxel centers. 

        
        Returns:
            (list): Minimum and maximum coordiantes for occupancy
            ref_3d (torch.tensor): Voxel centers

        """
        min_x = int((abs(-50 - self.occ_range[0]) + 0.5) / self.voxel_size[0])
        min_y = int((abs(-50 - self.occ_range[1]) + 0.5) / self.voxel_size[1])
        max_x = int((abs(50 - self.occ_range[0]) - 0.5) / self.voxel_size[0])
        max_y = int((abs(50 - self.occ_range[1]) - 0.5) / self.voxel_size[1])

        X = max_x - min_x + 1
        Y = max_y - min_y + 1
        Z = int((self.occ_range[5] - self.occ_range[2]) / self.voxel_size[2])

        xs = torch.arange(0, X).view(X, 1, 1).expand(X, Y, Z).type(torch.float32)
        ys = torch.arange(0, Y).view(1, Y, 1).expand(X, Y, Z).type(torch.float32)
        zs = torch.arange(0, Z).view(1, 1, Z).expand(X, Y, Z).type(torch.float32)

        ref_3d = torch.stack((xs, ys, zs), -1)
        ref_3d[..., 0] = (ref_3d[..., 0] + 0.5) * self.voxel_size[0] + self.occ_range[0]
        ref_3d[..., 1] = (ref_3d[..., 1] + 0.5) * self.voxel_size[1] + self.occ_range[1]
        ref_3d[..., 2] = (ref_3d[..., 2] + 0.5) * self.voxel_size[2] + self.occ_range[2]

        return (min_x, min_y, max_x, max_y), ref_3d

    def sample_points_triplane(self, triplane, points):
        """Sample features for points in triplane. 

        Args:
            triplane (list[torch.tensor]): Triplane
            points (torch.tensor): Points
            
        Returns:
            sampled_feat (torch.tensor): Triplane feature for each point

        """
        voxel_coors = torch.zeros_like(points)
        voxel_coors[..., 0] = (points[..., 0] - self.triplane_range[0]) / self.triplane_voxel_size[0]
        voxel_coors[..., 1] = (points[..., 1] - self.triplane_range[1]) / self.triplane_voxel_size[1]
        voxel_coors[..., 2] = (points[..., 2] - self.triplane_range[2]) / self.triplane_voxel_size[2]

        triplane_size = self.point_triplane_projector.grid_size
        voxel_coors[..., 0] = voxel_coors[..., 0] / (triplane_size[0] / 2) - 1
        voxel_coors[..., 1] = voxel_coors[..., 1] / (triplane_size[1] / 2) - 1
        voxel_coors[..., 2] = voxel_coors[..., 2] / (triplane_size[2] / 2) - 1

        b, h, w, d, p = voxel_coors.shape
        voxel_coors = voxel_coors.view(b, h, w*d, p)


        xy_feat = F.grid_sample(triplane[0], voxel_coors[..., [0, 1]], mode='bilinear', padding_mode='zeros')
        yz_feat = F.grid_sample(triplane[1], voxel_coors[..., [1, 2]], mode='bilinear', padding_mode='zeros')
        xz_feat = F.grid_sample(triplane[2], voxel_coors[..., [0, 2]], mode='bilinear', padding_mode='zeros')

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
        triplane (list): Triplane
        out_dir (str): Save directory
        count (int): Number to save

    """
    from sklearn.decomposition import PCA
    # triplane = triplane.permute(0, 1, 3, 4, 2).detach().cpu().squeeze().numpy()

    plt.rcParams['figure.figsize'] = [24, 12]
    fig, axes = plt.subplots(
            nrows=3, ncols=1
        )
    for row in range(3):
        axes[row].axis("off")


    for i in range(3):
        # Reshape the BEV to 2D array (128*128, 32)
        triplane_i = triplane[i].squeeze().permute(1, 2, 0).cpu().numpy()
        bev_reshaped = triplane_i.reshape(-1, triplane_i.shape[2])

        # Apply PCA to reduce to 3 components
        pca = PCA(n_components=3)
        bev_pca = pca.fit_transform(bev_reshaped)

        # Reshape back to 2D spatial dimensions (128, 128, 3)
        bev_pca_img = bev_pca.reshape(triplane_i.shape[0], triplane_i.shape[1], 3)

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
   

def evaluation_semantic(pred_occ, gt_occ, class_num):
    """Evaluation for semantic occupancy. 

    Args:
        pred_occ (torch.tensor): Predicted occupancy
        gt_occ (torch.tensor): Ground truth occupancy
        class_num (int): Number of classes
        
    Returns:
        results (list): IoU scores

    """
    results = []
   
    for i in range(pred_occ.shape[0]):
        gt_i, pred_i = gt_occ[i].cpu(), pred_occ[i].cpu()
        mask = (gt_i != 255)
        score = torch.zeros((class_num, 3))
        for j in range(class_num):
            if j == 0: #class 0 for geometry IoU
                score[j][0] += ((gt_i[mask] != 0) * (pred_i[mask] != 0)).sum()
                score[j][1] += (gt_i[mask] != 0).sum()
                score[j][2] += (pred_i[mask] != 0).sum()
            else:
                score[j][0] += ((gt_i[mask] == j) * (pred_i[mask] == j)).sum()
                score[j][1] += (gt_i[mask] == j).sum()
                score[j][2] += (pred_i[mask] == j).sum()

        
        results.append(score)
    
    results = torch.stack(results, dim=0)
    return results
    