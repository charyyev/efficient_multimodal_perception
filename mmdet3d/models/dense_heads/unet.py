import torch
import torch.nn as nn
import torch.nn.functional as F
import segmentation_models_pytorch as smp
from mmcv.cnn import build_conv_layer

from mmdet.models import HEADS

@HEADS.register_module()
class Unet(nn.Module):
    """Unet with resnet backbone. Takes 3D voxel features as input and predicts elevation in BEV 

    Args:
        input_dim (int): Input feature size
        height_dim (int): Height dimension of voxel features
        pad (int): Inpud padding

    """
    def __init__(self, 
                 input_dim, 
                 height_dim, 
                 pad, 
                 train_cfg = None, 
                 test_cfg = None):
        super().__init__()
        self.pad = pad
        self.proj = nn.Sequential(
            build_conv_layer(
                dict(type='Conv3d', bias=False),
                in_channels=input_dim,
                out_channels=1,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.ReLU(inplace=True)
        )

        self.decoder = smp.Unet(
            encoder_name="resnet34",             
            in_channels=height_dim,
            encoder_weights = None,                 
            classes=1,                     
            )

    def forward(self, x):
        """Forward call. 

        Args:
            x (torch.tensor): Input
            
        Returns:
            x (torch.tensor): Features

        """
        x = self.proj(x).squeeze(1)
        x = x.permute(0, 3, 1, 2)
        B, C, H, W = x.shape
        x = F.pad(x, (self.pad, self.pad, self.pad, self.pad))
        
        x = self.decoder(x)
        x = x[:, :, self.pad:self.pad + H, self.pad:self.pad + W]
        return x
        

    def loss(self, pred, target, mask):
        """Compute loss. 

        Args:
            pred (torch.tensor): Predicted semantic occupancy
            target (torch.tensor): Target semantic occupancy
            mask (torch.tensor): Which part to mask (used for sparse supervision)
            
        Returns:
            loss_dict (dict): Computed loss in dictionary

        """
        loss_dict = {}
        loss = F.mse_loss(pred * mask, target * mask, reduction="none")
        loss_dict["loss"] =  loss.sum() / (mask.sum() + 1e-8)

        return loss_dict
        

