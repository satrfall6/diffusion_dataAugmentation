# -*- coding: utf-8 -*-
"""
Created on Fri Mar  8 14:49:22 2024

@author: s4503302
"""

import os 
import numpy as np 
import h5py
from scipy.io import loadmat, savemat
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import trange
os.chdir(Path(__file__).parents[0])
root_path = Path.cwd()
from scipy.interpolate import interp1d, UnivariateSpline

from CONFOCAL_IMAGING_LEI.confocal_imaging import DMAS_updated
#%%
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

if __name__ == '__main__':
    num_partitions = 10
    if_random = False
    if not if_random:
        fixed = '_fixed' # ['', '_fixed']
    else:
        fixed = '_random'
    
    for par in range(4,num_partitions+1):
        FDTD_h5 = root_path / f'data/FDTD_stroke{fixed}_{par}.h5'
        if not if_random:
            FDTD_empty_h5 = root_path / f'data/FDTD_empty{fixed}.h5'
        else:
            FDTD_empty_h5 = root_path / f'data/FDTD_empty{fixed}_{par}.h5'
        with h5py.File(FDTD_h5) as FDTD, h5py.File(FDTD_empty_h5) as FDTD_empty:
            '''
            FDTD = h5py.File(FDTD_h5, 'r')
            FDTD_empty = h5py.File(FDTD_empty_h5, 'r')
            '''
            # print(FDTD.keys())
            fdtd_key_list = list(FDTD.keys())
            fdtd_empty_key_list = list(FDTD_empty.keys())
    
            for i in trange(len(fdtd_key_list)):
                # print(np.array(FDTD[fdtd_key_list[i]]).shape)
                # print(np.array(FDTD_empty[fdtd_empty_key_list[i]]).shape)
                fdtd_key = fdtd_key_list[i]
                if not if_random:
                    fdtd_empty_key = fdtd_empty_key_list[0]
                else:
                    fdtd_empty_key = fdtd_key
    
                fdtd_tr_signal = np.array(FDTD[fdtd_key]) - np.array(FDTD_empty[fdtd_empty_key])
                
        
                fdtd_tr_signal = interpolate_array(fdtd_tr_signal, 1200)
        
                confocal_result = DMAS_updated(fdtd_tr_signal.reshape(256,-1), if_sii=True)
                plt.imshow(confocal_result)
                plt.title(fdtd_key+'_cofocal')
                plots_dir = f'./data/plots/{fixed[1:]}_head/permittivity'
                if not os.path.exists(plots_dir):
                    os.makedirs(plots_dir)
                plt.tight_layout()
                plt.savefig(f'{plots_dir}/{fdtd_key}_cofocal.png')
                plt.close()
                #%% for showing signals ,tt, tr
                # fdtd_signal = interpolate_array(np.array(FDTD[fdtd_key]),256).reshape(256,-1)
                # fdtd_empty_signal = interpolate_array(np.array(FDTD_empty[fdtd_empty_key]),256).reshape(256,-1)
                # fdtd_tr_signal = interpolate_array(fdtd_tr_signal, 256).reshape(256,-1)
                # fig, ax = plt.subplots(4, 1, figsize=(12, 12))
                # ax[0].imshow(fdtd_signal)
                # ax[0].set_title('total')
                # ax[1].imshow(fdtd_empty_signal)
                # ax[1].set_title('empty')
                # ax[2].imshow(fdtd_tr_signal)
                # ax[2].set_title('tr')
                # ax[3].imshow(confocal_result)
                # ax[3].set_title('confocal')
                # plt.show()
