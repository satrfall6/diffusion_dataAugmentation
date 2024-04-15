# -*- coding: utf-8 -*-
"""
Created on Tue Jun 13 14:14:04 2023

@author: User
"""

from fnmatch import fnmatch
import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import h5py
import numpy as np
import torchvision.transforms as transforms

#%%

class MinMaxChannelNormalization(nn.Module):
    def __init__(self, max_vals=None):
        super().__init__()
        self.max_vals = max_vals # whapes [256,1]

    def forward(self, x):
        # Use precomputed min and max if provided, else calculate
        max_vals = self.max_vals


        x_normalized = (x + max_vals)  / (2 * max_vals) *2 - 1
        return x_normalized
    
class Self_MinMaxChannelNormalization(nn.Module):
    def __init__(self, max_vals=None):
        super().__init__()
        self.max_vals = max_vals

    def forward(self, x):
        # Use precomputed min and max if provided, else calculate
        max_vals = self.max_vals


        x_normalized = (x / max_vals) 
        return x_normalized
    
class random_head_diffusion_loader_reduce_ch(Dataset):
    def __init__(self, total_h5_paths, healthy_h5_paths, loc_label_path, reduced_dim_idx,
                 norm_values_tt=None, norm_values_tr=None, if_aligned=True):



        self.total_data = [(path, key) for path in total_h5_paths for key in self._read_h5_keys(path)]

        self.healthy_data = [(path, key) for path in healthy_h5_paths for key in self._read_h5_keys(path)]
        self.loc_label_path = loc_label_path
        self.reduced_dim_idx = reduced_dim_idx
        self.if_aligned = if_aligned
        # this part for testing 
        if norm_values_tt is not None and norm_values_tr is not None:
            self.norm_values_tt = norm_values_tt
            self.norm_values_tr = norm_values_tr
        else:
            self.norm_values_tt, self.norm_values_tr = self._read_min_max(self.total_data, self.healthy_data)
            # only for training 
            self.norm_values_tr = self.norm_values_tr.reshape(16,16,-1)[reduced_dim_idx]
            self.norm_values_tt = self.norm_values_tt.reshape(16,16,-1)[reduced_dim_idx]
            # print(f'fucking norm shape:{self.norm_values_tr}')
        self.minMax_transform_tt = transforms.Compose([
                                                    MinMaxChannelNormalization(self.norm_values_tt),
                                                    transforms.Normalize((0.4822), (0.247)),

                                                    # transforms.Resize(32)
                                                    # Add other transformations here if needed
                                                ])
        self.minMax_transform_tr = transforms.Compose([
                                                    MinMaxChannelNormalization(self.norm_values_tr),
                                                    transforms.Normalize((0.4822), (0.247)),
                                                    # transforms.Resize(32)
                                                    # Add other transformations here if needed
                                                ])
        
    def _read_h5_keys(self, file_path):
        with h5py.File(file_path, 'r') as h5_file:
            return list(h5_file.keys())

    def _read_min_max(self, total_data, healthy_data):
        # Create an empty array to store the distinct min and max values
        max_values_tt = np.full((256), -np.inf)
        max_values_tr = np.full((256), -np.inf)
        for path, key in total_data:
            '''
            new_path_str = str(path).replace("empty", "new_string")
            '''
            path_healty = str(path).replace("stroke", "empty")
            with h5py.File(path, 'r') as h5_file, h5py.File(path_healty, 'r') as h5_healthy:
                     
                # Iterate over the dictionary keys to find min and max values
                data_slice = np.abs(np.array(h5_file[key]).reshape(-1,64))  # Get the data corresponding to the key
                max_slice = np.max(data_slice, axis=(1))  # Find max over 1st and 3rd dimensions
                max_values_tt = np.maximum(max_values_tt, max_slice)  # Update max values
    
                # Iterate over the dictionary keys to find min and max values
                data_slice = np.abs((np.array(h5_file[key])-np.array(h5_healthy[key])).reshape(-1,64))  # Get the data corresponding to the key
                max_slice = np.max(data_slice, axis=(1))  # Find max over 1st and 3rd dimensions
                max_values_tr = np.maximum(max_values_tr, max_slice)  # Update max values
    
        return torch.tensor(max_values_tt).unsqueeze(1), torch.tensor(max_values_tr).unsqueeze(1)

    def _get_item_from_files(self, data_list, index):
        
        corr_h5_path = data_list[index][0] # data list are tuples in [[path_of_partition, keys_of_exp],..]
        key_exp = data_list[index][1]
        with h5py.File(corr_h5_path, 'r') as h5_file:
            item = np.array(h5_file[key_exp])
        return item
    
    def _get_loc_label(self, loc_label_path, key):
        with h5py.File(loc_label_path, 'r') as h5_file:
            loc_label = np.array(h5_file[key])
        return loc_label
    def _get_type_label(self, key):
        
        '''
        exmple of key : 'Exp07999_HAE_random-head-v0'
        '''
        stroke_type = key.split('_')[1]
        if stroke_type == 'ISC':
            type_label = 0
        elif stroke_type == 'HAE':
            type_label = 1
        else: # might need no stroke or assertion for error detection
            pass
        return type_label
    
    def __getitem__(self, index):
        domain_total = self._get_item_from_files(self.total_data, index)
        # print('*'*50)
        # print(f'fucking first loaded signal shape: {domain_total.shape}')
        # print('*'*50)
        domain_total = torch.tensor(domain_total[self.reduced_dim_idx]).float().unsqueeze(0)
        # print(f'fucking loaded signal shape: {domain_total.shape}')
        # print('*'*50)
        # domain_total = torch.tensor(domain_total, dtype=torch.float32).view(-1,domain_total.shape[-1]).unsqueeze(0)

        if self.if_aligned:
            domain_healthy = self._get_item_from_files(self.healthy_data, index)
        else:
            random_healthy_idx = np.random.randint(0, self.total_healthy_samples)
            domain_healthy = self._get_item_from_files(self.healthy_data, random_healthy_idx)

        # domain_healthy = torch.tensor(domain_healthy, dtype=torch.float32).view(-1,256).unsqueeze(0)
        domain_healthy = torch.tensor(domain_healthy[self.reduced_dim_idx]).float().unsqueeze(0)

        loc_label = self._get_loc_label(self.loc_label_path, self.total_data[index][1]) # 0 is path lf .h5
        type_label = self._get_type_label(self.total_data[index][1]) # 0 is path lf .h5
        

        tr = self.minMax_transform_tr(torch.abs(domain_total-domain_healthy))
        tt = self.minMax_transform_tt(torch.abs(domain_total))
        
        # print(f'now tr shapes: {tr.shape}')
        # self_norm_transform_tr = transforms.Compose([
        #                                     Self_MinMaxChannelNormalization(torch.max(tr,2)[0].permute(1,0)),
        #                                     transforms.Normalize((0.4822), (0.247)),

        #                                             ])
        # tr = self_norm_transform_tr(tr)
        # print(f'after self transform shape: {tr.shape}')
        # tr /= tr.max()
        # tt /= tt.max()
        
        return {'tr': tr, 'tr_keys': self.total_data[index][1],
                'tt': tt, 'tt_keys': self.total_data[index][1],
                'loc_label': loc_label, 'type_label': type_label}

    def __len__(self):
        return len(self.total_data)    
    
class random_head_diffusion_loader(Dataset):
    def __init__(self, total_h5_paths, healthy_h5_paths, loc_label_path,
                 norm_values_tt=None, norm_values_tr=None, if_aligned=True):



        self.total_data = [(path, key) for path in total_h5_paths for key in self._read_h5_keys(path)]

        self.healthy_data = [(path, key) for path in healthy_h5_paths for key in self._read_h5_keys(path)]
        self.loc_label_path = loc_label_path

        self.if_aligned = if_aligned
        if norm_values_tt is not None and norm_values_tr is not None:
            self.norm_values_tt = norm_values_tt
            self.norm_values_tr = norm_values_tr
        else:
            self.norm_values_tt, self.norm_values_tr = self._read_min_max(self.total_data, self.healthy_data)

        self.minMax_transform_tt = transforms.Compose([
                                                    MinMaxChannelNormalization(self.norm_values_tt),
                                                    # transforms.Resize(32)
                                                    # Add other transformations here if needed
                                                ])
        self.minMax_transform_tr = transforms.Compose([
                                                    MinMaxChannelNormalization(self.norm_values_tr),
                                                    # transforms.Resize(32)
                                                    # Add other transformations here if needed
                                                ])
        
    def _read_h5_keys(self, file_path):
        with h5py.File(file_path, 'r') as h5_file:
            return list(h5_file.keys())

    def _read_min_max(self, total_data, healthy_data):
        # Create an empty array to store the distinct min and max values
        max_values_tt = np.full((256), -np.inf)
        max_values_tr = np.full((256), -np.inf)
        for path, key in total_data:
            '''
            new_path_str = str(path).replace("empty", "new_string")
            '''
            path_healty = str(path).replace("stroke", "empty")
            with h5py.File(path, 'r') as h5_file, h5py.File(path_healty, 'r') as h5_healthy:
                     
                # Iterate over the dictionary keys to find min and max values
                data_slice = np.abs(np.array(h5_file[key]).reshape(-1,256))  # Get the data corresponding to the key
                max_slice = np.max(data_slice, axis=(1))  # Find max over 1st and 3rd dimensions
                max_values_tt = np.maximum(max_values_tt, max_slice)  # Update max values
    
                # Iterate over the dictionary keys to find min and max values
                data_slice = np.abs((np.array(h5_file[key])-np.array(h5_healthy[key])).reshape(-1,256))  # Get the data corresponding to the key
                max_slice = np.max(data_slice, axis=(1))  # Find max over 1st and 3rd dimensions
                max_values_tr = np.maximum(max_values_tr, max_slice)  # Update max values
    
        return torch.tensor(max_values_tt).unsqueeze(1), torch.tensor(max_values_tr).unsqueeze(1)

    def _get_item_from_files(self, data_list, index):
        
        corr_h5_path = data_list[index][0] # data list are tuples in [[path_of_partition, keys_of_exp],..]
        key_exp = data_list[index][1]
        with h5py.File(corr_h5_path, 'r') as h5_file:
            item = np.array(h5_file[key_exp])
        return item
    
    def _get_loc_label(self, loc_label_path, key):
        with h5py.File(loc_label_path, 'r') as h5_file:
            loc_label = np.array(h5_file[key])
        return loc_label
    def _get_type_label(self, key):
        
        '''
        exmple of key : 'Exp07999_HAE_random-head-v0'
        '''
        stroke_type = key.split('_')[1]
        if stroke_type == 'ISC':
            type_label = 0
        elif stroke_type == 'HAE':
            type_label = 1
        else: # might need no stroke or assertion for error detection
            pass
        return type_label
    
    def __getitem__(self, index):
        domain_total = self._get_item_from_files(self.total_data, index)
        domain_total = torch.tensor(domain_total, dtype=torch.float32).view(-1,256).unsqueeze(0)

        if self.if_aligned:
            domain_healthy = self._get_item_from_files(self.healthy_data, index)
        else:
            random_healthy_idx = np.random.randint(0, self.total_healthy_samples)
            domain_healthy = self._get_item_from_files(self.healthy_data, random_healthy_idx)

        domain_healthy = torch.tensor(domain_healthy, dtype=torch.float32).view(-1,256).unsqueeze(0)
        loc_label = self._get_loc_label(self.loc_label_path, self.total_data[index][1]) # 0 is path lf .h5
        type_label = self._get_type_label(self.total_data[index][1]) # 0 is path lf .h5
        
        tr = self.minMax_transform_tr(torch.abs(domain_total-domain_healthy))
        tt = self.minMax_transform_tt(torch.abs(domain_total))

        return {'tr': tr, 'tr_keys': self.total_data[index][1],
                'tt': tt, 'tt_keys': self.total_data[index][1],
                'loc_label': loc_label, 'type_label': type_label}

    def __len__(self):
        return len(self.total_data)