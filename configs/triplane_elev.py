_base_ = [
    'nuscenes_range.py',
    'cosine_2x.py',
    'default_runtime.py',
]

# -------------------model--------------------
voxel_size = (0.4, 0.4, 0.1)
volume = (100, 100, 80)
triplane_range = [-25, -25, -5, 25, 25, 3]
pc_range = [-20, -20, -5, 20, 20, 3]

range_img_size = (32, 1024)
patch_size = (1, 4)
masking_ratio_range = 0.0

img_size = (256, 512)
masking_ratio_img = 0.0

ckpt_path = "/cluster/scratch/scharyyev/thesis/rmae/triplane-orig/latest.pth"


model = dict(
    type='TriplaneElev',
    encoder=dict(
        type='JointEncoder',
        lidar_encoder=dict(
            type="MaskConvNeXt",
            arch="tiny",
            drop_path_rate=0.2,
            out_indices=(3),
            norm_out=True,
            frozen_stages=1,
            in_channels = 193,
            stem_patch_size = (1, 4),
            # init_cfg=dict(
            #     type="Pretrained",
            #     checkpoint="data/ckpts/convnextS_1kpretrained_official_style.pth",
            # ),
            mae_cfg=dict(
                downsample_scale=8, downsample_dim=768, mask_ratio=masking_ratio_img, learnable=False
            ),
            downsample_height = False,
        ),

        camera_encoder=dict(
            type="MaskConvNeXtV2",
            arch="tiny",
            drop_path_rate=0.2,
            out_index=1,
            norm_out=True,
            frozen_stages=1,
            stem_patch_size = (2, 2),
            # init_cfg=dict(
            #     type="Pretrained",
            #     checkpoint="data/ckpts/convnextS_1kpretrained_official_style.pth",
            # ),
            mae_cfg=dict(
                downsample_scale=8, downsample_dim=768, mask_ratio=masking_ratio_img, learnable=False
            ),
            ),
        ),
        
    neck = dict(
        type="MixVisionTransformer",
        img_size=(128, 32),
        patch_size=1,
        embed_dim = 1024
    ),
    
    decoder=dict(
        type='Unet',
        input_dim = 32,
        height_dim=volume[2],
        pad = 14
    ),
    ckpt_path = ckpt_path,
    volume = volume,
    voxel_size = voxel_size,
    pc_range = pc_range,
    triplane_range = triplane_range
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
    warmup_iters=1000,
    warmup_ratio=1.0 / 10,
    min_lr_ratio=1e-7)

momentum_config = None

# runtime settings
epochs = 50
find_unused_parameters = True
runner = dict(type='EpochBasedRunner', max_epochs=epochs)
evaluation = dict(interval=epochs+1)  # Don't evaluate when doing pretraining
workflow = [("train", 1)]
checkpoint_config = dict(
    interval=5,
    max_keep_ckpts=5,
)

# log_config = dict(
#     hooks=[
#         dict(type='TextLoggerHook'),
#         dict(type='WandbLoggerHook',
#             init_kwargs={'project': 'thesis',
#                          'name': "triplane_elev_full"},
#             interval=50)
#     ]
# )


fp16 = dict(loss_scale=32.0)
data = dict(
    samples_per_gpu=1,
    workers_per_gpu=1,
)
