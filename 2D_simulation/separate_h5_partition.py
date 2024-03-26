# -*- coding: utf-8 -*-
"""
Created on Mon Mar 11 09:57:53 2024

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
#%%
# write to h5
if_random = False
if not if_random:
    fixed = '_fixed' # ['', '_fixed']
else:
    fixed = '_random'

stroke_per_path = root_path / f'data/stroke_per{fixed}.h5'
  
stroke_con_path = root_path / f'data/stroke_con{fixed}.h5'        

empty_per_path = root_path / f'data/empty_per{fixed}.h5'

empty_con_path = root_path / f'data/empty_con{fixed}.h5'

num_partitions = 10
num_cases = 20000
cases_in_partition = int(num_cases / num_partitions) # divide into 10 partitions
def write_to_h5(stroke_per_path, stroke_con_path, empty_per_path, empty_con_path,
                mask_per, mask_con, empty_mask_per, empty_mask_con, exp_id):
                
    
    with h5py.File(stroke_per_path, 'a')  as stroke_per_h5, h5py.File(stroke_con_path, 'a')  as stroke_con_h5,\
        h5py.File(empty_per_path, 'a') as empty_per_h5, h5py.File(empty_con_path, 'a') as empty_con_h5:
    
        stroke_per_h5.create_dataset(exp_id, data=mask_per.astype(np.float64))
        stroke_con_h5.create_dataset(exp_id, data=mask_con.astype(np.float64))
        empty_per_h5.create_dataset(exp_id, data=empty_mask_per.astype(np.float64))
        empty_con_h5.create_dataset(exp_id, data=empty_mask_con.astype(np.float64))
#%%
with h5py.File(stroke_per_path, 'a')  as stroke_per_h5, h5py.File(stroke_con_path, 'a')  as stroke_con_h5,\
    h5py.File(empty_per_path, 'a') as empty_per_h5, h5py.File(empty_con_path, 'a') as empty_con_h5:
        
        key_list = list(stroke_per_h5.keys())
        
        for par in trange(1, num_partitions+1):
            stroke_per_path_par = root_path / f'data/stroke_per{fixed}_{par}.h5'            
            stroke_con_path_par = root_path / f'data/stroke_con{fixed}_{par}.h5'   
            empty_per_path_par = root_path / f'data/empty_per{fixed}_{par}.h5'
            empty_con_path_par = root_path / f'data/empty_con{fixed}_{par}.h5'
            start_idx = (par-1)*cases_in_partition

            for i in range(start_idx, start_idx+cases_in_partition):
                key = key_list[i]
                
                mask_per = np.array(stroke_per_h5[key])
                mask_con = np.array(stroke_con_h5[key])
                empty_mask_per = np.array(empty_per_h5[key])
                empty_mask_con = np.array(empty_con_h5[key])
            
                write_to_h5(stroke_per_path_par, stroke_con_path_par, empty_per_path_par, empty_con_path_par,
                                mask_per, mask_con, empty_mask_per, empty_mask_con, key)