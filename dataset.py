# -*- coding: utf-8 -*-
"""
Created on Tue Jun 13 14:14:04 2023

@author: User
"""

from fnmatch import fnmatch
import os
import torch
from torch.utils.data import Dataset
import h5py
import numpy as np
import torchvision.transforms as transforms
import random

#%%

class random_head_diffusion_loader(Dataset):
    def __init__(self, total_h5_paths, healthy_h5_paths, split='train', split_ratio=0.7,
                 transforms_A=None, transforms_B=None, if_aligned=False):


        # Load data keys from the .h5 files
        # self.target_data = [(path, self._read_h5_keys(path)) for path in target_h5_paths]
        self.total_data = {i: self._read_h5_keys(path) for i, path in enumerate(total_h5_paths)}
        self.total_path_list = [path for path in total_h5_paths]

        # self.healthy_data = [(path, self._read_h5_keys(path)) for path in healthy_h5_paths]
        self.healthy_data = {i: self._read_h5_keys(path) for i, path in enumerate(healthy_h5_paths)}
        self.healthy_path_list = [path for path in healthy_h5_paths]
        
        # self.total_target_samples = sum(len(keys) for _, keys in self.target_data)
        self.total_total_samples = sum(len(self.total_data[key]) for key in self.total_data.keys())
        # self.total_healthy_samples = sum(len(keys) for _, keys in self.healthy_data)
        self.healthy_data_samples = sum(len(self.healthy_data[key]) for key in self.healthy_data.keys())

        self.if_aligned = if_aligned
        self.transform_A = transforms.Compose(transforms_A) if transforms_A else None
        self.transform_B = transforms.Compose(transforms_B) if transforms_B else None

    def _read_h5_keys(self, file_path):
        with h5py.File(file_path, 'r') as h5_file:
            return list(h5_file.keys())

    def _get_item_from_files(self, data_list, index):
        for file_path, keys in data_list:
            if index < len(keys):
                with h5py.File(file_path, 'r') as h5_file:
                    item = np.array(h5_file[keys[index]])
                return item
            index -= len(keys)
        raise IndexError("List index out of range")
        
    def _get_item_direct(self, index):
        num_samples_per_file = 2000  # Assuming each file has 2000 samples
        file_index = index // num_samples_per_file
        key_index = index % num_samples_per_file
    
        # Construct the file name based on the file index
        # Assuming file names are like 'partition_1.h5', 'partition_2.h5', etc.
        file_name = f"partition_{file_index + 1}.h5"
        key_name = f"xxxx_{key_index:04d}"
    
        with h5py.File(f"{self.base_path}/{file_name}", 'r') as h5_file:
            item = np.array(h5_file[key_name])
        return item
    
    def __getitem__(self, index):
        domain_target = self._get_item_from_files(self.target_data, index)
        if self.transform_A:
            domain_target = self.transform_A(torch.tensor(domain_target, dtype=torch.float32).unsqueeze(0))

        if self.if_aligned:
            domain_healthy = self._get_item_from_files(self.healthy_data, index)
        else:
            random_healthy_idx = np.random.randint(0, self.total_healthy_samples)
            domain_healthy = self._get_item_from_files(self.healthy_data, random_healthy_idx)

        if self.transform_B:
            domain_healthy = self.transform_B(torch.tensor(domain_healthy, dtype=torch.float32).unsqueeze(0))

        return {'A': domain_target, 'A_keys': index, 'B': domain_healthy, 'B_keys': random_healthy_idx if not self.if_aligned else index}

    def __len__(self):
        return max(self.total_target_samples, self.total_healthy_samples)