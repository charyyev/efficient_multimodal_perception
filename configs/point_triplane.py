_base_ = [
    'nuscenes_surf_sam.py',
    'cosine_2x.py',
    'default_runtime.py',
]

# -------------------model--------------------
voxel_size = (0.4, 0.4, 0.1)
point_cloud_range = [-25, -25, -5, 25, 25, 3]
grid_size = [128, 128, 80]

img_size = (256, 512)

model = dict(
    type='PointTriplane',
    point_triplane_projector=dict(
        type="PointTriplaneProjector",
        grid_size = grid_size,
        base_channels=128,
        split=[25,25,20],
        in_channels = 5,
        out_channels = 128

    ),

    camera_encoder=dict(
        type="MaskConvNeXt",
        arch="tiny",
        drop_path_rate=0.2,
        out_indices=(3),
        norm_out=True,
        frozen_stages=1,
        stem_patch_size = (2, 2),
        # init_cfg=dict(
        #     type="Pretrained",
        #     checkpoint="/cluster/scratch/scharyyev/thesis/models/ckpts/convnextS_1kpretrained_official_style.pth",
        # ),
        ),
    
    triplane_encoder=dict(
        type="MaskConvNeXt",
        arch="tiny",
        drop_path_rate=0.2,
        out_indices=(3),
        norm_out=True,
        frozen_stages=1,
        stem_patch_size = (1, 1),
        out_all_scale = True,
        in_channels = 128,
        # init_cfg=dict(
        #     type="Pretrained",
        #     checkpoint="/cluster/scratch/scharyyev/thesis/models/ckpts/convnextS_1kpretrained_official_style.pth",
        # ),
        ),
    
    fpn=dict(
        type='GeneralizedLSSFPN',
        in_channels=[96, 192, 384, 768],
        out_channels=96,
        start_level=0,
        num_outs=3,
        norm_cfg=dict(
          type='BN2d',
          requires_grad=True,
          track_running_stats=True),
        act_cfg=dict(
          type='ReLU',
          inplace=True),
        upsample_cfg=dict(
          mode='bilinear',
          align_corners=False),
    ),
    camera_decoder=dict(
        type='MixVisionTransformerHead',
        img_size = (256, 512),
        patch_size=(7, 7),
        in_chans=96,
        embed_dim=96,
        norm_pix_loss=False,
        actual_patch_size = (4, 4),
        img_in_chans = 3
    ),
    
    surface_decoder=dict(
        type='InterpNet',
        latent_size=96, 
        out_channels=1, 
        radius=1.0,   
        n_non_manifold_pts=2048, 
        non_manifold_dist=0.1
    ),
    contrastive = False,
    voxel_size = voxel_size,
    pc_range = point_cloud_range

)


# optimizer
lr = 2.5e-4  # max learning rate
optimizer = dict(
    _delete_=True,
    type='AdamW',
    lr=lr,
    betas=(0.95, 0.99),  # the momentum is change during training
    weight_decay=0.001)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))

lr_config = dict(
    _delete_=True,
    policy='CosineAnnealing',
    warmup='linear',
    warmup_iters=20,
    warmup_ratio=1.0 / 10,
    min_lr_ratio=1e-1)

momentum_config = None

# runtime settings
epochs = 50
find_unused_parameters = True
runner = dict(type='EpochBasedRunner', max_epochs=epochs)
evaluation = dict(interval=epochs+1)  # Don't evaluate when doing pretraining
workflow = [("train", 1)]
checkpoint_config = dict(
    interval=5,
    max_keep_ckpts=1000,
)

log_config = dict(
    interval = 50,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='WandbLoggerHook',
            init_kwargs={'project': 'thesis',
                         'name': "point_triplane_surf_cam"})
    ]
)


fp16 = dict(loss_scale=32.0)
data = dict(
    samples_per_gpu=1,
    workers_per_gpu=4,
)
