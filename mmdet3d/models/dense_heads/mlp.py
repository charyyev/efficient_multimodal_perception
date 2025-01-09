import torch
import torch.nn as nn
import torch.nn.functional as F
import segmentation_models_pytorch as smp
from mmcv.cnn import build_conv_layer

from mmdet.models import HEADS

@HEADS.register_module()
class Mlp(nn.Module):
    """Mlp for 3D voxels. 

    Args:
        input_dim (int): Input dimension
        num_classes (int): Number of classes

    """
    def __init__(self, 
                 input_dim,
                 num_classes, 
                 train_cfg = None, 
                 test_cfg = None):
        super().__init__()

        self.conv1 = nn.Sequential(
            build_conv_layer(
                dict(type='Conv3d', bias=False),
                in_channels=input_dim,
                out_channels=2 * input_dim,
                kernel_size=1,
                stride=1
            ),
            nn.ReLU(inplace=True)
        )

        self.conv2 = nn.Sequential(
            build_conv_layer(
                dict(type='Conv3d', bias=False),
                in_channels=2 * input_dim,
                out_channels= input_dim,
                kernel_size=1
            ),
            nn.ReLU(inplace=True)
        )

        self.conv3 = nn.Sequential(
            build_conv_layer(
                dict(type='Conv3d', bias=False),
                in_channels=input_dim,
                out_channels= num_classes,
                kernel_size=1
            )
        )

        

    def forward(self, x):
        """Forward call. 

        Args:
            x (torch.tensor): Input
            
        Returns:
            x (torch.tensor): Features

        """
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
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
        

