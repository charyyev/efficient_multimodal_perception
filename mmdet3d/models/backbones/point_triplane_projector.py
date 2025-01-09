import torch.nn as nn
import torch
import torch.nn.functional as F
from mmdet.models.builder import BACKBONES

import torch_scatter
import numpy as np
from spconv.pytorch import SparseConvTensor, SparseMaxPool3d


@BACKBONES.register_module()
class PointTriplaneProjector(nn.Module):
    """Projects point features to triplane. 

        Args:
            grid_size (list): Voxel size.
            in_channels (int): Number of channels to extract pointwise features.
            out_channels (int): Output feature size.
            base_channels (int): Base channels
            split (list): Group size of spatial pooling
            track_running_stats (bool): Whether to track stats
            
        """
    def __init__(self, grid_size, in_channels=10, out_channels=256, 
                 base_channels=32, split=[4,4,4], track_running_stats=True):
        super(PointTriplaneProjector, self).__init__()
        self.grid_size = grid_size
        self.split = split
        
        # point-wise mlp
        self.point_mlp = nn.Sequential(
            nn.BatchNorm1d(in_channels, track_running_stats=track_running_stats),

            nn.Linear(in_channels, 64),
            nn.BatchNorm1d(64, track_running_stats=track_running_stats),
            nn.ReLU(),

            nn.Linear(64, 128),
            nn.BatchNorm1d(128, track_running_stats=track_running_stats),
            nn.ReLU(),

            nn.Linear(128, 256),
            nn.BatchNorm1d(256, track_running_stats=track_running_stats),
            nn.ReLU(),

            nn.Linear(256, out_channels)
        )

        self.reduce_cam_channels = nn.Linear(768, out_channels)
        
        # sparse max pooling
        
        self.pool_xy = SparseMaxPool3d(kernel_size=[1,1,int(self.grid_size[2]/split[2])],
                                              stride=[1,1,int(self.grid_size[2]/split[2])], padding=0)
        self.pool_yz = SparseMaxPool3d(kernel_size=[int(self.grid_size[0]/split[0]),1,1],
                                              stride=[int(self.grid_size[0]/split[0]),1,1], padding=0)
        self.pool_xz = SparseMaxPool3d(kernel_size=[1,int(self.grid_size[1]/split[1]),1],
                                              stride=[1,int(self.grid_size[1]/split[1]),1], padding=0)
        
        in_channels = [int(base_channels * s) for s in split]
        out_channels = [int(base_channels) for s in split]
        self.mlp_xy = nn.Sequential(nn.Linear(in_channels[2], out_channels[2]), nn.ReLU(), nn.Linear(out_channels[2], out_channels[2]))
        self.mlp_yz = nn.Sequential(nn.Linear(in_channels[0], out_channels[0]), nn.ReLU(), nn.Linear(out_channels[0], out_channels[0]))
        self.mlp_xz = nn.Sequential(nn.Linear(in_channels[1], out_channels[1]), nn.ReLU(), nn.Linear(out_channels[1], out_channels[1]))

    def forward(self, points, grid_ind, cam_point_features):
        """Forward call. 

        Args:
            points (torch.tensor): Point coordinates.
            grid_ind (torch.tensor): Voxel index of each point
            cam_point_features (torch.tensor): pointwise image features
            
        Returns:
            list: Triplane

        """
        device = points[0].get_device()

        cat_pt_ind, cat_pt_fea = [], []
        for i_batch in range(len(grid_ind)):
            cat_pt_ind.append(F.pad(grid_ind[i_batch], (1, 0), 'constant', value=i_batch))
            cat_pt_fea.append(points[i_batch][:, 0:5])

        cat_pt_ind = torch.cat(cat_pt_ind, dim=0)
        cat_pt_fea = torch.cat(cat_pt_fea, dim = 0)
        cat_cam_pt_fea = torch.cat(cam_point_features, dim = 0)
        cat_cam_pt_fea = self.reduce_cam_channels(cat_cam_pt_fea)

        pt_num = cat_pt_ind.shape[0]

        # shuffle the data
        shuffled_ind = torch.randperm(pt_num, device=device)
        cat_pt_fea = cat_pt_fea[shuffled_ind, :]
        cat_pt_ind = cat_pt_ind[shuffled_ind, :]
        cat_cam_pt_fea = cat_cam_pt_fea[shuffled_ind, :]
        
        # unique xy grid index
        unq, unq_inv, unq_cnt = torch.unique(cat_pt_ind, return_inverse=True, return_counts=True, dim=0)
        unq = unq.type(torch.int64)

        # process feature
        processed_cat_pt_fea = self.point_mlp(cat_pt_fea) + cat_cam_pt_fea
        pooled_data = torch_scatter.scatter_max(processed_cat_pt_fea, unq_inv, dim=0)[0]

        processed_pooled_data = pooled_data
        
        # sparse conv & max pooling
        coors = unq.int()
        batch_size = coors[-1][0] + 1
        ret = SparseConvTensor(processed_pooled_data, coors, np.array(self.grid_size), batch_size)
        # ret = self.spconv(ret)
        tpv_xy = self.mlp_xy(self.pool_xy(ret).dense().permute(0,2,3,4,1).flatten(start_dim=3)).permute(0,3,1,2)
        tpv_yz = self.mlp_yz(self.pool_yz(ret).dense().permute(0,3,4,2,1).flatten(start_dim=3)).permute(0,3,1,2)
        tpv_xz = self.mlp_xz(self.pool_xz(ret).dense().permute(0,2,4,3,1).flatten(start_dim=3)).permute(0,3,1,2)

        return [tpv_xy, tpv_yz, tpv_xz]