import torch
import torch.nn as nn
import torch.nn.functional as F
import segmentation_models_pytorch as smp
from mmcv.cnn import build_conv_layer

from mmdet.models import HEADS

@HEADS.register_module()
class PointMlp(nn.Module):
    """Mlp that operate on pointwise features. 

    Args:
        laten_size (int): Input dimension
        out_channels (int): Number of classes

    """
    def __init__(self, 
                 latent_size,
                 out_channels, 
                 train_cfg = None, 
                 test_cfg = None):
        super().__init__()
        mlp_layers = [torch.nn.Linear(latent_size, latent_size) for _ in range(2)]
        self.mlp_layers = nn.ModuleList(mlp_layers)
        self.fc_out = torch.nn.Linear(latent_size, out_channels)
        self.activation = torch.nn.ReLU()
        

    def forward(self, x):
        """Forward call. 

        Args:
            x (torch.tensor): Input
            
        Returns:
            x (torch.tensor): Features

        """
        for l in self.mlp_layers:
            x = l(self.activation(x))
        x = self.fc_out(x)
        return x
        

    def loss(self, pred, target):
        """Compute loss. 

        Args:
            pred (torch.tensor): Predicted semantic occupancy
            target (torch.tensor): Target semantic occupancy
            
        Returns:
            loss_dict (dict): Computed loss in dictionary

        """
        loss_dict = {}
        loss = F.cross_entropy(pred, target, ignore_index=255)
        loss_dict["loss"] =  loss

        return loss_dict
        

