# -*- coding: utf-8 -*-
"""
Created on Wed Apr  3 10:02:32 2024

@author: s4503302
"""

import os 
import numpy as np
import matplotlib.pyplot as plt
import h5py
from pathlib import Path
from scipy.interpolate import interp1d
from fnmatch import fnmatch
from tqdm import tqdm, trange

from CONFOCAL_IMAGING_LEI.confocal_imaging import DMAS_updated

os.chdir(Path(__file__).parents[0])
root_path = Path.cwd()

def interpolate_array(array, new_size):
    """
    Interpolates a 3D array from its current size to a new size.

    Parameters:
    array (numpy.ndarray): The original array of shape [16, 16, n].
    new_size (int): The desired first dimension size of the new array.

    Returns:
    numpy.ndarray: The interpolated array of shape [16, 16, new_size].
    """
    h, w, n = array.shape
    original_indices = np.linspace(0, n-1, n)
    new_indices = np.linspace(0, n-1, new_size)
    new_array = np.zeros((h, w, new_size))

    # Check if there are enough points for cubic interpolation
    if n >= 4:  # Cubic spline requires at least 4 points
        kind = 'cubic'
    else:  # Fallback to linear if not enough points
        kind = 'linear'

    for i in range(h):
        for j in range(w):
            interp_func = interp1d(original_indices, array[i, j, :], kind=kind, fill_value="extrapolate")
            new_array[i, j, :] = interp_func(new_indices)

    return new_array
#%% load dataset
dim_reduce_to = 64
dataset_path = root_path / 'data'
new_dataset_path = root_path / 'data_64'
if not os.path.exists(new_dataset_path):
    os.makedirs(new_dataset_path) 
if_random = True

if if_random:
    total_h5_paths = [path \
                 for path in list(Path(dataset_path).rglob("FDTD*random*.h5"))\
                 if fnmatch(path.parts[-1], '*stroke*')
                 and not 'empty' in path.parts[-1]
                 and not 'per' in path.parts[-1]
                 and not 'con' in path.parts[-1]]
                 

    healthy_h5_paths = [path \
                 for path in list(Path(dataset_path).rglob("FDTD*random*.h5"))\
                 if fnmatch(path.parts[-1], '*empty*')
                 and not 'stroke' in path.parts[-1]
                 and not 'per' in path.parts[-1]
                 and not 'con' in path.parts[-1]]
else:
    total_h5_paths = [path \
                 for path in list(Path(dataset_path).rglob("FDTD*fixed*.h5"))\
                 if fnmatch(path.parts[-1], '*stroke*')
                 and not 'empty' in path.parts[-1]
                 and not 'per' in path.parts[-1]
                 and not 'con' in path.parts[-1]]
                 

    healthy_h5_paths = [path \
                 for path in list(Path(dataset_path).rglob("FDTD*fixed*.h5"))\
                 if fnmatch(path.parts[-1], '*empty*')
                 and not 'stroke' in path.parts[-1]
                 and not 'per' in path.parts[-1]
                 and not 'con' in path.parts[-1]]

#%%
for i in trange(len(total_h5_paths)):
    
    h5_file = total_h5_paths[i]
    if if_random:
        h5_empty_file = healthy_h5_paths[i]
        assert h5_file.name.split('_')[-1] == h5_empty_file.name.split('_')[-1]
    else:
        h5_empty_file = healthy_h5_paths[0]
        
    new_h5_file = new_dataset_path / h5_file.name
    new_empty_h5_file = new_dataset_path / h5_empty_file.name

    with h5py.File(h5_file, 'r') as fdtd_h5, h5py.File(h5_empty_file, 'r') as fdtd_empyt_h5,\
        h5py.File(new_h5_file, 'a') as dim_reduced_fdtd_h5, h5py.File(new_empty_h5_file, 'a') as dim_reduced_fdtd_empty_h5:
            
        fdtd_key_list = list(fdtd_h5.keys())
        fdtd_empty_key_list = list(fdtd_empyt_h5.keys())
        
        for k in range(len(fdtd_key_list)):
            fdtd_key = fdtd_key_list[k]
            if not if_random: # if fixed we only have one empty
                fdtd_empty_key = fdtd_empty_key_list[0]
                if i == 0:
                    reduced_dim_empty_td = interpolate_array(np.array(fdtd_empyt_h5[fdtd_empty_key]), dim_reduce_to)
                    dim_reduced_fdtd_empty_h5.create_dataset(fdtd_empty_key, data=reduced_dim_empty_td)
            else: # for random boundary, we have to add new emtpy everytime
                fdtd_empty_key = fdtd_key
                reduced_dim_empty_td = interpolate_array(np.array(fdtd_empyt_h5[fdtd_empty_key]), dim_reduce_to)
                dim_reduced_fdtd_empty_h5.create_dataset(fdtd_empty_key, data=reduced_dim_empty_td)

            reduced_dim_td = interpolate_array(np.array(fdtd_h5[fdtd_key]), dim_reduce_to)
            
            dim_reduced_fdtd_h5.create_dataset(fdtd_key, data=reduced_dim_td)
            
            # # use confocal to check if dim reduction influence the results
            # print(reduced_dim_td.shape)
            # print(reduced_dim_empty_td.shape)

            # fdtd_tr_signal = reduced_dim_td - reduced_dim_empty_td
            # fdtd_tr_signal = interpolate_array(fdtd_tr_signal, 1200)
            
            # confocal_result = DMAS_updated(fdtd_tr_signal.reshape(256,-1), if_sii=True)
            # plt.imshow(confocal_result)
            # plt.title(fdtd_key+'_reduced_cofocal')
            # plt.show()
            # break
            # ##########################
