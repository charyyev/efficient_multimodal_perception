# ---------------------------------------------------------------
# Copyright (c) 2021, NVIDIA Corporation. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# ---------------------------------------------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial

from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import math
from mmdet.models import NECKS


class Mlp(nn.Module):
    """MixFFN module. 

    Args:
        in_features (int): Input channels
        hidden_features (int): Hidden channels
        out_features (int): Output channels
        act_layer (nn.Module): Activation layer

    """
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        """Forward call. 

        Args:
            x (torch.tensor): Input
            H (int): Height of image
            W (int): Width of image
            
        Returns:
            x (torch.tensor): Features

        """
        x = self.fc1(x)
        x = self.dwconv(x, H, W)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    """Efficient attention layer. 

    Args:
        dim (int): Input dimension
        num_heads (int): Number of heads
        qkv_bias (bool): Bias for query, key and value
        qk_scale (int): Value to scale query and key
        attn_drop (float): Drop rate for attention
        proj_drop (float): Drop rate for projection
        sr_ratio (int): Sequence reduction ratio 

    """
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        """Forward call. 

        Args:
            x (torch.tensor): Input
            H (int): Height of image
            W (int): Width of image
            
        Returns:
            x (torch.tensor): Features

        """
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        if self.sr_ratio > 1:
            x_ = x.permute(0, 2, 1).reshape(B, C, H, W)
            x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1)
            x_ = self.norm(x_)
            kv = self.kv(x_).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        else:
            kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class Block(nn.Module):
    """Mix Vision Transformer block. 

    Args:
        dim (int): Input dimension
        num_heads (int): Number of heads
        mlp_ratio (int): Ratio of hiddent dimension of MLP
        qkv_bias (bool): Bias for query, key and value
        qk_scale (int): Value to scale query and key
        drop (float): Drop rate
        attn_drop (float): Drop rate for attention
        drop_path (float): Drop path rate
        act_layer (nn.Module): Activation layer
        norm_layer (nn.Module): Norm layer
        sr_ratio (int): Sequence reduction ratio 
    """
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, sr_ratio=1):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, sr_ratio=sr_ratio)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        """Forward call. 

        Args:
            x (torch.tensor): Input
            H (int): Height of image
            W (int): Width of image
            
        Returns:
            x (torch.tensor): Features

        """
        x = x + self.drop_path(self.attn(self.norm1(x), H, W))
        x = x + self.drop_path(self.mlp(self.norm2(x), H, W))

        return x


class OverlapPatchEmbed(nn.Module):
    """Image to Patch Embedding. 

    Args:
        img_size (int): Image size
        patch_size (int): Patch size
        stride (int): Stride
        in_chans (int): Input channel
        embed_dim (int): Dimension for embedding
            
    """

    def __init__(self, img_size=224, patch_size=7, stride=4, in_chans=3, embed_dim=768):
        super().__init__()
        patch_size = to_2tuple(patch_size)

        self.img_size = img_size
        self.patch_size = patch_size
        self.H, self.W = img_size[0] // patch_size[0], img_size[1] // patch_size[1]
        self.num_patches = self.H * self.W
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride,
                              padding=(patch_size[0] // 2, patch_size[1] // 2))
        self.norm = nn.LayerNorm(embed_dim)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        """Forward call. 

        Args:
            x (torch.tensor): Input
            
        Returns:
            x (torch.tensor): Pathces
            H (int): Height of image
            W (int): Width of image

        """
        x = self.proj(x)
        _, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)

        return x, H, W



@NECKS.register_module()
class MixVisionTransformer(nn.Module):
    """Efficient transfomer block that maps image and range image features to triplane. 

    Args:
        img_size (tuple): Image size
        patch_size (int): Patch size
        in_chans (int): Input channel
        embed_dim (int): Dimension for embedding
        qkv_bias (bool): Bias for query, key and value
        qk_scale (int): Value to scale query and key
        drop_rate (float): Drop rate
        attn_drop_rate (float): Drop rate for attention
        norm_layer (nn.Module): Norm layer
            
    """
    def __init__(self, img_size=(128, 32), patch_size=1, in_chans=768, embed_dim=768,
                 qkv_bias=False, qk_scale=None, drop_rate=0.,
                 attn_drop_rate=0., norm_layer=nn.LayerNorm):
        super().__init__()

        # patch_embed
        self.patch_embed = OverlapPatchEmbed(img_size=img_size, patch_size=patch_size, stride=1, in_chans=in_chans,
                                              embed_dim=embed_dim)
        

        # transformer encoder
        depths = 2
        self.transformer_block = nn.ModuleList([Block(
            dim=embed_dim, num_heads=4, mlp_ratio=2, qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, norm_layer=norm_layer,
            sr_ratio=1)
            for i in range(depths)])
        self.norm1 = norm_layer(embed_dim)

        self.pixel_shuffle = nn.PixelShuffle(upscale_factor = 2)
        self.conv1 = nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(128, 96, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()


    def freeze_patch_emb(self):
        self.patch_embed1.requires_grad = False

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed1', 'pos_embed2', 'pos_embed3', 'pos_embed4', 'cls_token'}  # has pos_embed may be better


    def forward(self, x):
        """Forward call. 

        Args:
            x (torch.tensor): Input
            
        Returns:
            x (torch.tensor): Features

        """
        B = x.shape[0]

        x, H, W = self.patch_embed(x)
        
        dim = int(math.sqrt(H * W))
        assert dim == math.sqrt(H * W)

        for i, blk in enumerate(self.transformer_block):
            x = blk(x, H, W)
        x = self.norm1(x)
        x = x.reshape(B, dim, dim, -1).permute(0, 3, 1, 2).contiguous()

        x = self.pixel_shuffle(x)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)

        return x

@NECKS.register_module()
class MixVisionTransformerHigh(nn.Module):
    """Efficient transfomer block that maps image and range image features to triplane, longer range. 

    Args:
        img_size (tuple): Image size
        patch_size (int): Patch size
        in_chans (int): Input channel
        embed_dim (int): Dimension for embedding
        qkv_bias (bool): Bias for query, key and value
        qk_scale (int): Value to scale query and key
        drop_rate (float): Drop rate
        attn_drop_rate (float): Drop rate for attention
        norm_layer (nn.Module): Norm layer
            
    """
    def __init__(self, img_size=(128, 32), patch_size=1, in_chans=768, embed_dim=768,
                 qkv_bias=False, qk_scale=None, drop_rate=0.,
                 attn_drop_rate=0., norm_layer=nn.LayerNorm):
        super().__init__()

        # patch_embed
        self.patch_embed = OverlapPatchEmbed(img_size=img_size, patch_size=patch_size, stride=1, in_chans=in_chans,
                                              embed_dim=embed_dim)


        # transformer encoder
        depths = 2
        self.transformer_block = nn.ModuleList([Block(
            dim=embed_dim, num_heads=4, mlp_ratio=2, qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, norm_layer=norm_layer,
            sr_ratio=1)
            for i in range(depths)])
        self.norm1 = norm_layer(embed_dim)

        self.pixel_shuffle = nn.PixelShuffle(upscale_factor = 2)
        self.conv1 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(128, 96, kernel_size=3, stride=1, padding=1)
        self.conv_transpose = nn.ConvTranspose2d(256, 256, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.relu = nn.ReLU()

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()


    def freeze_patch_emb(self):
        self.patch_embed1.requires_grad = False

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed1', 'pos_embed2', 'pos_embed3', 'pos_embed4', 'cls_token'}  # has pos_embed may be better


    def forward(self, x):
        """Forward call. 

        Args:
            x (torch.tensor): Input
            
        Returns:
            x (torch.tensor): Features

        """
        B = x.shape[0]

        x, H, W = self.patch_embed(x)

        dim = int(math.sqrt(H * W))
        assert dim == math.sqrt(H * W)

        for i, blk in enumerate(self.transformer_block):
            x = blk(x, H, W)
        x = self.norm1(x)
        x = x.reshape(B, dim, dim, -1).permute(0, 3, 1, 2).contiguous()

        x = self.pixel_shuffle(x)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv_transpose(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.conv3(x)

        return x



class DWConv(nn.Module):
    """Depthwise separable convolution. 

    Args:
        dim (torch.tensor): Input dimension

    """
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x, H, W):
        """Forward call. 

        Args:
            x (torch.tensor): Input
            H (int): Height of image
            W (int): Width of image
            
        Returns:
            x (torch.tensor): Pathces

        """
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)

        return x