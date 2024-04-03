# -*- coding: utf-8 -*-
"""
Created on Tue Feb 20 11:40:32 2024

@author: User
"""
import sys

import os 
import random
import numpy as np
import h5py
import torch
from torch import optim
from torchvision import datasets, transforms, utils
from torch.utils.data import DataLoader
# from denoising_diffusion_pytorch import Unet, GaussianDiffusion
from denoising_diffusion_pytorch.classifier_free_guidance import Unet, GaussianDiffusion
# from denoising_diffusion_pytorch.classifier_free_guidance_locEmbed import Unet, GaussianDiffusion

import time
import matplotlib.pyplot as plt
import math
import cv2
from pathlib import Path
from fnmatch import fnmatch
from tqdm import tqdm
import wandb
import argparse
# root_path = Path(__file__).parents[0]
os.chdir(Path(__file__).parents[0])
root_path = Path.cwd()
# Add the folder containing your module to the system path
sys.path.append(str(root_path/'2D_simulation/CONFOCAL_IMAGING_LEI'))
sys.path.append(str(root_path/'2D_simulation'))

print('*'*30,f'Working in {root_path}','*'*30,)
print(root_path)

from dataset import random_head_diffusion_loader
from confocal_imaging import DMAS_updated
from verify_signal import interpolate_array
from generate_random_head import make_grid


parser = argparse.ArgumentParser()
parser.add_argument('--project_goal', type=str, default='Diffusion_tr', help='Project name for wandb')
parser.add_argument('--test_obj', type=str, default='type_location', help='name for separating wandb project')
parser.add_argument('--n_epochs', type=int, default=180, help='number of epochs of training')
parser.add_argument('--batchSize_train', type=int, default=10, help='size of the batches')
parser.add_argument('--batchSize_test', type=int, default=4, help='size of the batches')
parser.add_argument('--img_size', type=int, default=64, help='size of the batches')

parser.add_argument('--lr', type=float, default=0.00005, help='initial learning rate of G')

opt = parser.parse_args()
print(opt)

# initialize boundary for plotting
shape_1_size, shape_0_size = 8, 8
mask_con = np.zeros((512,512), dtype=np.float32)
mask_for_fill = np.zeros_like(mask_con)
# Draw fixed outer ellipse
cv2.ellipse(mask_for_fill, (256,256), (90,105), 0, 0, 360, 2, -1)
boundary = mask_for_fill.copy()
boundary[boundary!=0] = 255
boundary = cv2.resize(boundary, (256,256))
total_grids = make_grid(boundary, shape_1_size, shape_0_size)


def save_loss(loss_list):
    
    # After training, plot the average loss per epoch
    plt.figure(figsize=(10, 5))
    plt.plot(loss_list, label='Average Loss per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Curve')
    plt.legend()
    plt.savefig(f'training_loss_curve{opt.test_obj}.png')
    
def de_sym_norm(generated_signals, norm):
    return (generated_signals+1) / 2 * (2*norm) - norm

def check_results(generated_signals, norm, epoch, exp_id=None, save_dir=None):
    
    for i, b in enumerate(range(generated_signals.shape[0])):
        if i >=4:
            break
        
        generated_signal = generated_signals[b]
        if generated_signal.shape[0] != 256:
            denormed_signal = interpolate_array(generated_signal.reshape(16,16,-1), 256)
            denormed_signal = torch.tensor(denormed_signal).view(256,-1).detach()
        else: 
            denormed_signal = torch.tensor(generated_signal).clone().detach()
        denormed_signal = de_sym_norm(denormed_signal, norm)
        # denormed_signal = interpolate_array(denormed_signal.reshape(16,16,256), 1200)
        # confocal_result = DMAS_updated(denormed_signal.reshape(-1,1200), False)
        # print(denormed_signal.shape)
        
        confocal_result = DMAS_updated(denormed_signal, False)
            
    
        # Create a figure with 2 subplots
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))  # Adjust figure size as needed
    
        # Plot confocal_result
        axes[0].imshow(confocal_result, cmap='Spectral_r')  # Adjust colormap as needed
        axes[0].axis('off')
        axes[0].set_title('Confocal Result')
        
        # Plot generated siganl
        axes[1].imshow(generated_signal, cmap='gray')  # Adjust colormap as needed
        axes[1].axis('off')
        axes[1].set_title('generated signal')
        
        # Save the figure if needed
        if save_dir is not None:
            plt.savefig(save_dir / f'Epoch{epoch}_{exp_id[b]}.png')
            plt.close()
        else:
            plt.show()
#%% load dataset
dataset_path = root_path / '2D_simulation/data'
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
    loc_label_path = dataset_path / 'location_class_random.h5'
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
    loc_label_path = dataset_path / 'location_class_fixed.h5'
# Shuffle and split file paths
shuffle_seed = 1253
random.seed(shuffle_seed)
random.shuffle(total_h5_paths)
random.seed(shuffle_seed)
random.shuffle(healthy_h5_paths)

split_ratio = 0.7
num_train= int(len(total_h5_paths) * split_ratio)


train_total_paths = total_h5_paths[:num_train]
train_healthy_paths = healthy_h5_paths[:num_train]

test_total_paths = total_h5_paths[num_train:]
test_healthy_paths = healthy_h5_paths[num_train:]


train_dataset = random_head_diffusion_loader(train_total_paths, train_healthy_paths, loc_label_path)
# print(f'norm for tt is: {train_dataset.norm_values_tt}')
# print(f'norm for tr is: {train_dataset.norm_values_tr}')
test_dataset = random_head_diffusion_loader(test_total_paths, test_healthy_paths, loc_label_path,
                                            train_dataset.norm_values_tt, train_dataset.norm_values_tr)
train_loader = DataLoader(train_dataset, batch_size=opt.batchSize_train, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=opt.batchSize_test, shuffle=False)
# define norm array
norm = train_dataset.norm_values_tr

#%%
# for batch in tqdm(train_loader):
#     break
# generated_signal = batch['tr'].squeeze(1)
# loc_label = batch['loc_label']
# type_label = batch['type_label']

# check_results(generated_signal, loc_label , norm)
# plt.imshow(generated_signal[0,:,:].detach().cpu())
#%%
num_classes = 2 # [loc: 103, type:2]
model = Unet(
    dim = 64,
    channels = 1,
    dim_mults = (1, 2, 4, 8),
    num_classes = num_classes,
    cond_drop_prob = 0.5
).cuda()

diffusion = GaussianDiffusion(
    model,
    image_size = opt.img_size,
    timesteps = 500    # number of steps
).cuda()

#%%
wandb.init(project=f'{opt.project_goal}',
            group=f'{opt.test_obj}', reinit = True, config=opt)
# wandb.run.name = test_feature + '_' + test_model    
wandb.run.name = f'{opt.project_goal}_{opt.test_obj}'
optimizer = optim.Adam(model.parameters(), lr=opt.lr)
epoch_loss_list = []  # Store average loss per epoch
for epoch in range(opt.n_epochs ):
    #%%
    epoch_loss = 0.0
    for batch_idx, batch in enumerate(train_loader):
        
        signals = batch['tr'].float().cuda()
        # loc_label = batch['loc_label'].float().cuda()
        type_label = batch['type_label'].cuda()

        # loc_label = torch.sum(loc_label,1).to(torch.int)
    #%%
        optimizer.zero_grad()
        # loss = diffusion(signals, classes = loc_label)  # Forward pass and loss computation
        loss = diffusion(signals, classes = type_label)  # Forward pass and loss computation

        loss.backward()  # Backward pass
    
        optimizer.step()  # Update model parameters

        # Weight clipping
        for p in model.parameters():
            p.data.clamp_(-0.01, 0.01)  # Clipping the weights to the range [-0.01, 0.01]

        
        epoch_loss += loss.item()
        if batch_idx % 500 == 0:  # Log every 100 batches
            print(f'Epoch: {epoch}, Batch: {batch_idx}, Loss: {loss.item()}')
    #    wandb.log({"Batch_idx": batch_idx, "batch_loss": loss
    #               })
    # Calculate the average loss for the epoch
    avg_epoch_loss = epoch_loss / len(train_loader)
    epoch_loss_list.append(avg_epoch_loss)
    wandb.log({"epoch": epoch, "loss": avg_epoch_loss
                })
    print(f'Epoch: {epoch}, Average Loss: {avg_epoch_loss}')
    print('='*100)
    # Here you would save or display sampled_images, but remember they're on CUDA and need processing
    #%%
    if epoch % 10 == 0:
        # Simple timing for one sampling operation - place this where you need it
        start_time = time.time()


        for batch in tqdm(test_loader):
            # sample_classes = batch['loc_label'].float().cuda()
            sample_classes = batch['type_label'].cuda()

            exp_id = batch['tr_keys']
            
        # sample_classes = torch.randn([1,103]).cuda()
        sampled_images = diffusion.sample(
            classes = sample_classes,
            cond_scale = 6.                # condition scaling, anything greater than 1 strengthens the classifier free guidance. reportedly 3-8 is good empirically
        ).detach().cpu().squeeze(1)
        end_time = time.time()  # Capture the end time after the function call
        # Calculate the duration
        duration = end_time - start_time
        print(f"Sampling 1000 steps took {duration} seconds.")
        
        # plot checking and save plots
        plot_path = root_path / f'plots/{opt.test_obj}'
        if not os.path.exists(plot_path):
            os.makedirs(plot_path) 
        check_results(sampled_images, norm, epoch,
                      exp_id, save_dir=plot_path)

        final_weights_path = root_path / f'ckpts/{opt.test_obj}'
        if not os.path.exists(final_weights_path):
            os.makedirs(final_weights_path)
        # if loss_G < best_loss:
            # lr_not_update_count = 0
            # best_loss = loss_G
        torch.save(diffusion.state_dict(), final_weights_path/f'diffusion{opt.test_obj}.pth')
save_loss(epoch_loss_list)
#%%
# training_images = torch.rand(4, 3, 64, 64).cuda() # images are normalized from 0 to 1
# image_classes = torch.randint(0, num_classes, (8,)).cuda()    # say 10 classes
# loss = diffusion(training_images, classes = image_classes)
# loss.backward()

# # after a lot of training
# start_time = time.time()  
# diffusion.is_ddim_sampling = True
# print(f'if using DDIM sampling: {diffusion.is_ddim_sampling }')
# sampled_images = diffusion.sample(
#     classes = image_classes,
#     cond_scale = 6.                # condition scaling, anything greater than 1 strengthens the classifier free guidance. reportedly 3-8 is good empirically
# )
# print(sampled_images.shape) # (4, 3, 128, 128)
# end_time = time.time()  # Capture the end time after the function call
# # Calculate the duration
# duration = end_time - start_time
# print(f"Sampling 1000 steps took {duration} seconds.")
# show_images(sampled_images, 0)
# show_images(training_images, 0)

# interpolate_out = diffusion.interpolate(
#     training_images[:1],
#     training_images[:1],
#     image_classes[:1]
# )
# use ddim:19 sec/1000 steps; not using ddim: 18.7 sec/1000 steps

# diffusion.load_state_dict(torch.load(Path(f'{final_weights_path}/diffusion_cfg.pth')))

