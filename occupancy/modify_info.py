import pickle
import numpy as np
import os

""" Modify annotation info to add occupancy path
"""
if __name__ == "__main__":
    path = "/scratch/tmp.4677455.scharyyev/nuscenes-full/nuscenes_infos_train.pkl"
    save_path = "/scratch/tmp.4677455.scharyyev/nuscenes-full/nuscenes_occ_train.pkl"
    with open(path, "rb") as f:
        object = pickle.load(f)
        for i in range(len(object['infos'])):
            name = object['infos'][i]['lidar_path'].split("/")[-1] + ".npy"
            object['infos'][i]['occ_filename'] = "occupancy/" + name
        
        with open(save_path, "wb") as f:
            pickle.dump(object, f) 
            