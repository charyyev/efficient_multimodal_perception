from nuscenes.nuscenes import NuScenes

from nuscenes.utils.data_classes import LidarPointCloud
import numpy as np
import argparse
import os

voxel_size = 0.4
pointcloud_range = (-20, -20, -5, 20, 20, 3)
count = 0

def parse_args():
    parser = argparse.ArgumentParser(description='Data converter arg parser')
    parser.add_argument(
        '--dataroot',
        type=str,
        default='/cluster/scratch/scharyyev/thesis/nuscenes-13',
        help='specify the root path of dataset')
    parser.add_argument(
        '--save_path',
        type=str,
        default='/cluster/scratch/scharyyev/thesis/elevation-nuscenes-13',
        required=False,
        help='specify sweeps of lidar per example')
    
    args = parser.parse_args()
    return args

def generate_elevation_data(nusc: NuScenes, cur_sample, save_path):
    """Generates elevation for one sample. 

    Args:
        nusc (Nuscenes): Nuscenes object
        cur_sample (dict): Information about current sample
        save_path (str): Save location

    """
    global pointcloud_range
    global voxel_size
    global count

    lidar_data = nusc.get('sample_data',
                            cur_sample['data']['LIDAR_TOP'])
    if not os.path.exists(os.path.join(nusc.dataroot, lidar_data['filename'])):
        return
    pc = LidarPointCloud.from_file(os.path.join(nusc.dataroot, lidar_data['filename']))
    filename = os.path.split(lidar_data['filename'])[-1].split('.')[0]
    points = pc.points.T
    
    # Remove points of self vehicle
    radius = 2.0
    x_filt = np.abs(points[:, 0]) < radius
    y_filt = np.abs(points[:, 1]) < radius
    
    not_close = np.logical_not(np.logical_and(x_filt, y_filt))
    points = points[not_close]

    # remain points with a spatial range
    eps = 0.5
    mask = ((points[:, 0] > pointcloud_range[0] + eps) & (points[:, 1] > pointcloud_range[1] + eps) & (points[:, 2] > pointcloud_range[2] + eps) \
               & (points[:, 0] < pointcloud_range[3] - eps) & (points[:, 1] < pointcloud_range[4] - eps) & (points[:, 2] < pointcloud_range[5] - eps))
    
    points = points[mask]
    points_cp = np.copy(points)

    # create voxels and fill height values
    points = points[points[:, 2].argsort()] 

    points[:, 0] = (points[:, 0] - pointcloud_range[0]) / voxel_size
    points[:, 1] = (points[:, 1] - pointcloud_range[1]) / voxel_size
    
    pcd_xy = np.floor(points[:, 0:2]).astype(np.int)
    voxel = np.zeros((int((pointcloud_range[3] - pointcloud_range[0]) / voxel_size), int((pointcloud_range[4] - pointcloud_range[1]) / voxel_size)))
    mask = np.zeros((int((pointcloud_range[3] - pointcloud_range[0]) / voxel_size), int((pointcloud_range[4] - pointcloud_range[1]) / voxel_size)))
    voxel[pcd_xy[:, 0], pcd_xy[:, 1]] = points[:, 2]
    mask[pcd_xy[:, 0], pcd_xy[:, 1]] = 1

    np.savez(os.path.join(save_path, filename), elevation = voxel, mask = mask)
    count += 1
    




def convert2elevation(dataroot,
                        save_path):
    """Generates elevation for dataset. 

    Args:
        dataroot (str): Location of data
        save_path (str): Save location

    """
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    nusc = NuScenes(version='v1.0-trainval', dataroot=dataroot, verbose=True)
    for scene in nusc.scene:
        sample_token = scene['first_sample_token']
        cur_sample = nusc.get('sample', sample_token)
        while True:
            print(count)
            generate_elevation_data(nusc, cur_sample, save_path=save_path)
            
            if cur_sample['next'] == '':
                break
            cur_sample = nusc.get('sample', cur_sample['next'])
        

if __name__ == "__main__":
    args = parse_args()
    convert2elevation(args.dataroot, args.save_path)