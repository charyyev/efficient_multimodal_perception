from nuscenes.nuscenes import NuScenes

from nuscenes.utils.data_classes import LidarPointCloud
import numpy as np
import argparse
import os


count = 0
proj_fov_up = 10
proj_fov_down = -30
proj_W = 1024
proj_H = 32

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
        default='/cluster/scratch/scharyyev/thesis/nuscenes-13/samples/range_images',
        required=False,
        help='specify sweeps of lidar per example')
    
    args = parser.parse_args()
    return args

def generate_range_data(nusc: NuScenes, cur_sample):
    """Generates range image for one sample. 

    Args:
        nusc (Nuscenes): Nuscenes object
        cur_sample (dict): Information about current sample

    """

    global count
    global proj_fov_up
    global proj_fov_down
    global proj_H
    global proj_W

    lidar_data = nusc.get('sample_data',
                            cur_sample['data']['LIDAR_TOP'])
    if not os.path.exists(os.path.join(nusc.dataroot, lidar_data['filename'])):
        return
    pc = LidarPointCloud.from_file(os.path.join(nusc.dataroot, lidar_data['filename']))
    # filename = os.path.split(lidar_data['filename'])[-1].split('.')[0]
    filename = lidar_data['filename'].replace('LIDAR_TOP', 'RANGE_FULL')
    filename = filename.replace('.pcd.bin', '')
    points = pc.points.T
    
    # Remove points of self vehicle
    radius = 2.0
    x_filt = np.abs(points[:, 0]) < radius
    y_filt = np.abs(points[:, 1]) < radius
    
    not_close = np.logical_not(np.logical_and(x_filt, y_filt))
    points = points[not_close]

    
    # laser parameters
    fov_up = proj_fov_up / 180.0 * np.pi      # field of view up in rad
    fov_down = proj_fov_down / 180.0 * np.pi  # field of view down in rad
    fov = abs(fov_down) + abs(fov_up)  # get field of view total in rad

    # get depth of all points
    depth = np.linalg.norm(points[:, :3], 2, axis=1)

    # get scan components
    scan_x = points[:, 0]
    scan_y = points[:, 1]
    scan_z = points[:, 2]

    # get angles of all points
    yaw = np.arctan2(scan_x, scan_y)          #minus?
    pitch = np.arcsin(scan_z / (depth + 1e-8))

    # get projections in image coords
    proj_x = 0.5 * (yaw / np.pi + 1.0)          # in [0.0, 1.0]
    proj_y = 1.0 - (pitch + abs(fov_down)) / fov        # in [0.0, 1.0]

    # scale to image size using angular resolution
    proj_x *= proj_W                              # in [0.0, W]
    proj_y *= proj_H                              # in [0.0, H]

    # round and clamp for use as index
    proj_x = np.floor(proj_x)
    proj_x = np.minimum(proj_W - 1, proj_x)
    proj_x = np.maximum(0, proj_x).astype(np.int32)   # in [0,W-1]
    # proj_x = np.copy(proj_x)  # store a copy in orig order

    proj_y = np.floor(proj_y)
    proj_y = np.minimum(proj_H - 1, proj_y)
    proj_y = np.maximum(0, proj_y).astype(np.int32)   # in [0,H-1]
    # self.proj_y = np.copy(proj_y)  # stope a copy in original order

    # copy of depth in original order
    depth_cp = np.copy(depth)

    # order in decreasing depth
    indices = np.arange(depth.shape[0])
    order = np.argsort(depth)[::-1]
    depth = depth[order]
    indices = indices[order]
    points = points[order]
    
    proj_y = proj_y[order]
    proj_x = proj_x[order]
    
    range_img = np.zeros((proj_H, proj_W))
    range_points = np.zeros((proj_H, proj_W, 3))

    # assing to images
    range_img[proj_y, proj_x] = depth
    range_points[proj_y, proj_x] = points[:, :3]
    
    np.savez(os.path.join(nusc.dataroot, filename), range_image = range_img, range_points = range_points)
    count += 1

    




def convert2range(dataroot):
    """Generates range image for dataset. 

    Args:
        dataroot (str): Location of data

    """
    nusc = NuScenes(version='v1.0-trainval', dataroot=dataroot, verbose=True)
    for scene in nusc.scene:
        sample_token = scene['first_sample_token']
        cur_sample = nusc.get('sample', sample_token)
        while True:
            print(count)
            generate_range_data(nusc, cur_sample)
            
            if cur_sample['next'] == '':
                break
            cur_sample = nusc.get('sample', cur_sample['next'])
        

if __name__ == "__main__":
    args = parse_args()
    convert2range(args.dataroot)