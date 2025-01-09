# For nuScenes we usually do 10-class detection
class_names = [
    'car', 'truck', 'trailer', 'bus', 'construction_vehicle', 'bicycle',
    'motorcycle', 'pedestrian', 'traffic_cone', 'barrier'
]
dataset_type = 'NuScenesDataset'
data_root = '/cluster/scratch/scharyyev/thesis/nuscenes-13/'
elev_root = '/cluster/scratch/scharyyev/thesis/elevation-nuscenes-13/'
# data_root = ''
# Input modality for nuScenes dataset, this is consistent with the submission
# format which requires the information in input_modality.
input_modality = dict(
    use_lidar=True,
    use_camera=True,
    use_radar=False,
    use_map=False,
    use_external=False)
file_client_args = dict(backend='disk')
# Uncomment the following if use ceph or other file clients.
# See https://mmcv.readthedocs.io/en/latest/api.html#mmcv.fileio.FileClient
# for more details.
# file_client_args = dict(
#     backend='petrel',
#     path_mapping=dict({
#         './data/nuscenes/': 's3://nuscenes/nuscenes/',
#         'data/nuscenes/': 's3://nuscenes/nuscenes/'
#     }))

train_pipeline = [
    dict(
        type='LoadMultiViewImageFromFiles',
        to_float32=True,
    ),
    dict(
        type = 'LoadRangeImageFromFile'
    ),
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=11,
        use_dim=11,
        file_client_args=file_client_args
        ),
    dict(
        type='LoadOccGTFromFile',
        data_root = data_root
    ),
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True),
    dict(type='ImageAug3D', 
         final_dim=[256, 512],
         resize_lim=[0.44, 0.61],
         bot_pct_lim=[0.0, 0.0],
         rand_flip=True,
         is_train=True,
    ),
    dict(type='ImageNormalize', mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225],),
    # dict(type='ImageNormalize_mae', mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225],),
    dict(type='DefaultFormatBundle3D', class_names=class_names),
    dict(type='Collect3D', 
         keys=['range_image', 'img', 'gt_bboxes_3d', 'gt_labels_3d', "points"],
         meta_keys=['camera_intrinsics', 'camera2ego', 'lidar2ego',
                    'lidar2camera', 'camera2lidar', 'lidar2image', 
                    'img_aug_matrix', 'range_points', 'occupancy'],
         )
]
test_pipeline = [
   dict(
        type='LoadMultiViewImageFromFiles',
        to_float32=True,
    ),
    dict(
        type = 'LoadRangeImageFromFile'
    ),
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=11,
        use_dim=11,
        file_client_args=file_client_args
    ),
    dict(
        type='LoadOccGTFromFile',
        data_root = data_root
    ),
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True),
    dict(type='ImageAug3D', 
         final_dim=[256, 512],
         resize_lim=[0.44, 0.61],
         bot_pct_lim=[0.0, 0.0],
         rand_flip=False,
         is_train=False,
    ),

    dict(type='ImageNormalize', mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225],),
    # dict(type='ImageNormalize_mae', mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225],),
    dict(type='DefaultFormatBundle3D', class_names=class_names),
    dict(type='Collect3D', 
         keys=['range_image', 'img', 'gt_bboxes_3d', 'gt_labels_3d', "points"],
         meta_keys=['camera_intrinsics', 'camera2ego', 'lidar2ego',
                    'lidar2camera', 'camera2lidar', 'lidar2image', 
                    'img_aug_matrix', 'range_points', 'occupancy'],
         )
]
# construct a pipeline for data and gt loading in show function
# please keep its loading function consistent with test_pipeline (e.g. client)
eval_pipeline = [
    dict(
        type='LoadMultiViewImageFromFiles',
        to_float32=True,
    ),
    dict(
        type = 'LoadRangeImageFromFile'
    ),
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=11,
        use_dim=11,
        file_client_args=file_client_args
    ),
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True),
    dict(type='ImageNormalize', mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225],),
    # dict(type='ImageNormalize_mae', mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225],),
    dict(type='DefaultFormatBundle3D', class_names=class_names),
    dict(type='Collect3D', 
         keys=['range_image', 'img', 'gt_bboxes_3d', 'gt_labels_3d', 'range_points', "points"],
         meta_keys=['camera_intrinsics', 'camera2ego', 'lidar2ego',
                    'lidar2camera', 'camera2lidar', 'lidar2image', 
                    'img_aug_matrix'],
         )
]

data = dict(
    samples_per_gpu=4,
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_root + 'nuscenes_infos_train.pkl',
        pipeline=train_pipeline,
        classes=class_names,
        modality=input_modality,
        test_mode=False,
        # we use box_type_3d='LiDAR' in kitti and nuscenes dataset
        # and box_type_3d='Depth' in sunrgbd and scannet dataset.
        box_type_3d='LiDAR'),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_root + 'nuscenes_infos_val.pkl',
        pipeline=test_pipeline,
        classes=class_names,
        modality=input_modality,
        test_mode=False,
        box_type_3d='LiDAR'),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_root + 'nuscenes_infos_val.pkl',
        pipeline=test_pipeline,
        classes=class_names,
        modality=input_modality,
        test_mode=False,
        box_type_3d='LiDAR'))
# For nuScenes dataset, we usually evaluate the model at the end of training.
# Since the models are trained by 24 epochs by default, we set evaluation
# interval to be 24. Please change the interval accordingly if you do not
# use a default schedule.
evaluation = dict(interval=24, pipeline=eval_pipeline)
