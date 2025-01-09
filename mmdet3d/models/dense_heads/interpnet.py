import torch
import torch.nn as nn
import logging

from torch_geometric.nn import radius as search_radius, knn as search_knn, avg_pool_x
import torch.nn.functional as F

from functools import partial
from mmdet3d.models.builder import HEADS


@HEADS.register_module()
class InterpNet(torch.nn.Module):
    """Head for surface reconstruction. 

        Args:
            latent_size (torch.tensor): Size of features.
            out_channels (int): Number of classes
            radius (int): Radius for supervision
            n_non_manifold_pts (int): Number of support points
            non_manifold_dist (int): Delta, distance from point to be considered for query


    """
    def __init__(self, latent_size, out_channels, K=1, radius=1.0,  spatial_prefix="", 
                n_non_manifold_pts=None, non_manifold_dist=0.1
            ):
        super().__init__()

        self.out_channels = out_channels
        self.n_non_manifold_pts = n_non_manifold_pts
        self.non_manifold_dist = non_manifold_dist

        # layers of the decoder
        self.fc_in = torch.nn.Linear(latent_size+3, latent_size)
        mlp_layers = [torch.nn.Linear(latent_size, latent_size) for _ in range(2)]
        self.mlp_layers = nn.ModuleList(mlp_layers)
        self.fc_out = torch.nn.Linear(latent_size, self.out_channels)
        self.activation = torch.nn.ReLU()
        self.spatial_prefix = spatial_prefix

        self.radius = radius
        self.K = None
        self.search_function = partial(search_radius, r=self.radius)
        
    def forward(self, data):
        """Forward call. 

        Args:
            data (dict): Dictionary containing points and features.
            
        Returns:
            return_data: Dictionary of predictions and loss

        """
        
        pos_source = data["pos"]
        batch_source = data["latents_batch"]
    
        pos_target = data["pos_non_manifold"]
        batch_target = data["pos_non_manifold_batch"]
        latents = data["latents"]

        # neighborhood search
        row, col = self.search_function(x=pos_source, y=pos_target, batch_x=batch_source, batch_y=batch_target)

        # compute reltive position between query and input point cloud
        # and the corresponding latent vectors
        pos_relative = pos_target[row] - pos_source[col]
        latents_relative = latents[col]


        x = torch.cat([latents_relative, pos_relative], dim=1)

        # Decoder layers
        x = self.fc_in(x.contiguous())
        for i, l in enumerate(self.mlp_layers):
            x = l(self.activation(x))
        x = self.fc_out(x)

        return_data = {"predictions":x[:, 0],}

        if "occupancies" in data:
            occupancies = data["occupancies"][row]
            return_data["occupancies"] = occupancies

            #### Reconstruction loss
            recons_loss = F.binary_cross_entropy_with_logits(x[:,0], occupancies.float())
            return_data["surface_loss"] = recons_loss


        return return_data
    
    def test_forward(self, x):
        """Forward call for testing. 

        Args:
            x (torch.tensor): Input.
            
        Returns:
            torch.tensor: Predictions

        """

        x = self.fc_in(x.contiguous())
        for i, l in enumerate(self.mlp_layers):
            x = l(self.activation(x))
        x = self.fc_out(x)

        return x[:, 0]

        
    def create_targets(self, batched_points, feat):
        """Creates targets for supervision. 

        Args:
            batched_points (torch.tensor): Point coordinates in batch format.
            feat (torch.tensor): Triplane feature for each point
            
            
        Returns:
            data (dict): Dictionary containing points and features and targets.

        """
        if not isinstance(batched_points, list):
            mask = (batched_points==0).sum(3) != 3
            feat = feat.permute(0, 2, 3, 1)
            batch_size = batched_points.shape[0]
        else:
            batch_size = len(batched_points)

        data = {}
        batched_pos = []
        batched_pos_non_manifold = []
        batched_occupancy = []
        batched_latents = []

        batch_pos = []
        batch_pos_non_manifold = []

        for b in range(batch_size):
            if not isinstance(batched_points, list):
                points = batched_points[b][mask[b]]
                latents = feat[b][mask[b]]
            else:
                points = batched_points[b]
                latents = feat[b]

            # nmp -> non_manifold points
            n_nmp = self.n_non_manifold_pts

            # select the points for the current frame
            n_nmp_out = n_nmp // 3
            n_nmp_out_far = n_nmp // 3
            n_nmp_in = n_nmp - 2 * (n_nmp//3)
            nmp_choice_in = torch.randperm(points.shape[0])[:n_nmp_in]
            nmp_choice_out = torch.randperm(points.shape[0])[:n_nmp_out]
            nmp_choice_out_far = torch.randperm(points.shape[0])[:n_nmp_out_far]

            # center
            center = torch.zeros((1,3), dtype=torch.float, device = points.device)

            # in points
            pos = points[nmp_choice_in]
            dirs = F.normalize(pos, dim=1)
            pos_in = pos + self.non_manifold_dist * dirs * torch.rand((pos.shape[0],1), device = pos.device)
            occ_in = torch.ones(pos_in.shape[0], dtype=torch.long, device = pos.device)
            
            # out points
            pos = points[nmp_choice_out]
            dirs = F.normalize(pos, dim=1)
            pos_out = pos - self.non_manifold_dist * dirs * torch.rand((pos.shape[0],1), device = pos.device)
            occ_out = torch.zeros(pos_out.shape[0], dtype=torch.long, device = pos.device)

            # out far points
            pos = points[nmp_choice_out_far]
            dirs = F.normalize(pos, dim=1)
            pos_out_far = (pos - center) * torch.rand((pos.shape[0],1), device = pos.device) + center
            occ_out_far = torch.zeros(pos_out_far.shape[0], dtype=torch.long, device = pos.device)

            pos_non_manifold = torch.cat([pos_in, pos_out, pos_out_far], dim=0)
            occupancies = torch.cat([occ_in, occ_out, occ_out_far], dim=0)
            batch = b * torch.ones(occupancies.shape[0], dtype = torch.long, device = pos.device)
            pos_batch = b * torch.ones(points.shape[0], dtype = torch.long, device = pos.device)

            batched_pos_non_manifold.append(pos_non_manifold)
            batched_occupancy.append(occupancies)
            batched_pos.append(points)
            batched_latents.append(latents)

            batch_pos_non_manifold.append(batch)
            batch_pos.append(pos_batch)
            
           
        data["pos"] = torch.cat(batched_pos, dim = 0)
        data["occupancies"] = torch.cat(batched_occupancy, dim = 0)
        data["pos_non_manifold"] = torch.cat(batched_pos_non_manifold, dim = 0)
        data["latents"] = torch.cat(batched_latents, dim = 0)
        data["latents_batch"] = torch.cat(batch_pos, dim = 0)
        data["pos_non_manifold_batch"] = torch.cat(batch_pos_non_manifold, dim = 0)
        return data

        