# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
from typing import Callable, List, Union
import os

import numpy as np
from mmdet.datasets import DATASETS
from ..core.bbox import get_box_type
import mmcv
from .custom_3d import Custom3DDataset
import pdb


@DATASETS.register_module()
class WaymoDataset(Custom3DDataset):
    """Waymo Dataset.

    This class serves as the API for experiments on the Waymo Dataset.

    Please refer to `<https://waymo.com/open/download/>`_for data downloading.
    It is recommended to symlink the dataset root to $MMDETECTION3D/data and
    organize them as the doc shows.

    Args:
        data_root (str): Path of dataset root.
        ann_file (str): Path of annotation file.
        data_prefix (dict): data prefix for point cloud and
            camera data dict. Defaults to dict(
                                    pts='velodyne',
                                    CAM_FRONT='image_0',
                                    CAM_FRONT_LEFT='image_1',
                                    CAM_FRONT_RIGHT='image_2',
                                    CAM_SIDE_LEFT='image_3',
                                    CAM_SIDE_RIGHT='image_4')
        pipeline (List[dict]): Pipeline used for data processing.
            Defaults to [].
        modality (dict): Modality to specify the sensor data used
            as input. Defaults to dict(use_lidar=True).
        default_cam_key (str): Default camera key for lidar2img
            association. Defaults to 'CAM_FRONT'.
        box_type_3d (str): Type of 3D box of this dataset.
            Based on the `box_type_3d`, the dataset will encapsulate the box
            to its original format then converted them to `box_type_3d`.
            Defaults to 'LiDAR' in this dataset. Available options includes:

            - 'LiDAR': Box in LiDAR coordinates.
            - 'Depth': Box in depth coordinates, usually for indoor dataset.
            - 'Camera': Box in camera coordinates.
        load_type (str): Type of loading mode. Defaults to 'frame_based'.

            - 'frame_based': Load all of the instances in the frame.
            - 'mv_image_based': Load all of the instances in the frame and need
                to convert to the FOV-based data type to support image-based
                detector.
            - 'fov_image_based': Only load the instances inside the default
                cam, and need to convert to the FOV-based data type to support
                image-based detector.
        filter_empty_gt (bool): Whether to filter the data with empty GT.
            If it's set to be True, the example with empty annotations after
            data pipeline will be dropped and a random example will be chosen
            in `__getitem__`. Defaults to True.
        test_mode (bool): Whether the dataset is in test mode.
            Defaults to False.
        pcd_limit_range (List[float]): The range of point cloud
            used to filter invalid predicted boxes.
            Defaults to [-85, -85, -5, 85, 85, 5].
        cam_sync_instances (bool): If use the camera sync label
            supported from waymo version 1.3.1. Defaults to False.
        load_interval (int): load frame interval. Defaults to 1.
        max_sweeps (int): max sweep for each frame. Defaults to 0.
    """
    METAINFO = {
        'classes': ('Car', 'Pedestrian', 'Cyclist'),
        'palette': [
            (0, 120, 255),  # Waymo Blue
            (0, 232, 157),  # Waymo Green
            (255, 205, 85)  # Amber
        ]
    }

    def __init__(self,
                 data_root: str,
                 ann_file: str,
                 classes: None,
                 data_prefix: dict = dict(
                     pts='velodyne',
                     CAM_FRONT='image_0',
                     CAM_FRONT_LEFT='image_1',
                     CAM_FRONT_RIGHT='image_2',
                     CAM_SIDE_LEFT='image_3',
                     CAM_SIDE_RIGHT='image_4'),
                 pipeline: List[Union[dict, Callable]] = [],
                 modality: dict = dict(use_lidar=True),
                 default_cam_key: str = 'CAM_FRONT',
                 box_type_3d: str = 'LiDAR',
                 load_type: str = 'frame_based',
                 filter_empty_gt: bool = True,
                 test_mode: bool = False,
                 cam_sync_instances: bool = False,
                 load_interval: int = 1,
                 max_sweeps: int = 0,
                 **kwargs) -> None:
        self.load_interval = load_interval
        self.data_root = data_root
        # set loading mode for different task settings
        self.cam_sync_instances = cam_sync_instances
        # construct self.cat_ids for vision-only anns parsing
        self.cat_ids = range(len(self.METAINFO['classes']))
        self.cat2label = {cat_id: i for i, cat_id in enumerate(self.cat_ids)}
        self.max_sweeps = max_sweeps
        self.data_prefix = data_prefix
        # we do not provide backend_args to custom_3d init
        # because we want disk loading for info
        # while ceph loading for Prediction2Waymo

        super().__init__(
            data_root=data_root,
            ann_file=ann_file,
            pipeline=pipeline,
            classes=classes,
            modality=modality,
            box_type_3d=box_type_3d,
            filter_empty_gt=filter_empty_gt,
            test_mode=test_mode)

    

    def load_annotations(self, ann_file) -> List[dict]:
        """Add the load interval.

        Returns:
            list[dict]: A list of annotation.
        """  # noqa: E501
        # `self.ann_file` denotes the absolute annotation file path if
        # `self.root=None` or relative path if `self.root=/path/to/data/`.
        annotations = mmcv.load(ann_file)
        
        
        if not isinstance(annotations, dict):
            raise TypeError(f'The annotations loaded from annotation file '
                            f'should be a dict, but got {type(annotations)}!')
        if 'data_list' not in annotations or 'metainfo' not in annotations:
            raise ValueError('Annotation must have data_list and metainfo '
                             'keys')
        metainfo = annotations['metainfo']
        raw_data_list = annotations['data_list']
        
        raw_data_list = raw_data_list[::self.load_interval]
        

        return raw_data_list

    def get_data_info(self, index) -> Union[dict, List[dict]]:
        
        info = self.data_infos[index]
        info['timestamp'] /= 1e6
        if self.modality['use_lidar']:
            info['lidar_points']['lidar_path'] = \
                osp.join(
                    self.data_prefix.get('pts', ''),
                    info['lidar_points']['lidar_path'])

        
            info['pts_filename'] = os.path.join(self.data_root, "training", info['lidar_points']['lidar_path'])
            

        lidar2ego = np.eye(4).astype(np.float32)
        info["lidar2ego"] = lidar2ego

        if self.modality['use_camera']:
            info["image_paths"] = []
            info["lidar2camera"] = []
            info["lidar2image"] = []
            info["camera2ego"] = []
            info["camera_intrinsics"] = []
            info["camera2lidar"] = []
            for cam_id, img_info in info['images'].items():
                if 'img_path' in img_info:
                    if cam_id in self.data_prefix:
                        cam_prefix = self.data_prefix[cam_id]
                    else:
                        cam_prefix = self.data_prefix.get('img', '')
                    img_info['img_path'] = osp.join(cam_prefix,
                                                    img_info['img_path'])
                    
            for (cam_key, img_info) in info['images'].items():
                info["image_paths"].append(os.path.join(self.data_root, "training", img_info["img_path"]))
                
                info["lidar2camera"].append(np.array(img_info["lidar2cam"]))
                
                # camera intrinsics
                camera_intrinsics = np.eye(4).astype(np.float32)
                camera_intrinsics[:3, :3] = np.array(img_info["cam2img"])[:3, :3]
                info["camera_intrinsics"].append(camera_intrinsics)

                lidar2image = camera_intrinsics @ np.array(img_info["lidar2cam"])
                info["lidar2image"].append(lidar2image)

                camera2lidar = np.linalg.inv(np.array(img_info["lidar2cam"]))
                info["camera2lidar"].append(camera2lidar)
                info["camera2ego"].append(camera2lidar)

                
                
            return info