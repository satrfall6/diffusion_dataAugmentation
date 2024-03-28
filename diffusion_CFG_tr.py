# -*- coding: utf-8 -*-
"""
Created on Tue Feb 20 11:40:32 2024

@author: User
"""
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
import time
import matplotlib.pyplot as plt
import math
from copy import deepcopy
from pathlib import Path
from fnmatch import fnmatch
# root_path = Path(__file__).parents[0]
os.chdir(Path(__file__).parents[0])
root_path = Path.cwd()
print('*'*30,f'Working in {root_path}','*'*30,)
print(root_path)

from dataset import random_head_diffusion_loader


model_techs_sub = '_cfg'

def show_images(images, epoch):
    

    images_batch = images.detach().cpu().clone()
    batch_size = images_batch.shape[0]
    # Determine the grid size (rows x cols) for the subplot based on the batch size
    cols = int(math.sqrt(batch_size))
    rows = batch_size // cols + (batch_size % cols > 0)
    
    # Adjust cols and rows to fit the progression 2x2, 4x2, 4x4, 8x4, ...
    if rows * cols < batch_size:  # In case of perfect square numbers
        cols += 1
    if cols > rows and rows * cols >= batch_size * 2:
        # This adjustment helps maintain the grid's aspect ratio closer to square
        cols = cols // 2
        rows = rows * 2
    
    fig, axs = plt.subplots(rows, cols, figsize=(cols * 2.5, rows * 2.5))
    axs = axs.flatten()  # Flatten the array of axes for easy indexing

    for i, img in enumerate(images_batch):
        img = img
        img = img.permute(1, 2, 0)  # Change from (C, H, W) to (H, W, C)
        # img = img * torch.tensor([0.247, 0.243, 0.261]) + torch.tensor([0.4914, 0.4822, 0.4465])  # Unnormalize
        img = (img + 1) / 2
        img = img.clamp(0, 1)  # Ensure the image values are between 0 and 1
        axs[i].imshow(img)
        axs[i].axis('off')  # Hide the axes

    # Hide any unused subplot areas
    for i in range(len(images_batch), len(axs)):
        axs[i].axis('off')

    
    save_dir = Path('./Plots/')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    fname = save_dir/f'random_batch_epoch{model_techs_sub}_{epoch}.jpg'

    # plt.imshow(superimposed_img)
    fig.savefig(fname, bbox_inches='tight', pad_inches=0, dpi=150)
    
    # plt.tight_layout()
    # plt.show()
def save_loss(loss_list):
    
    # After training, plot the average loss per epoch
    plt.figure(figsize=(10, 5))
    plt.plot(loss_list, label='Average Loss per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Curve')
    plt.legend()
    plt.savefig(f'training_loss_curve{model_techs_sub}.png')


#%% load dataset
dataset_path = root_path / '2D_simulation/data'
if_random = True
from dataset import random_head_diffusion_loader

if if_random:
    total_h5_paths = [path \
                 for path in list(Path(dataset_path).rglob("*random*.h5"))\
                 if fnmatch(path.parts[-1], '*stroke*')
                 and not 'empty' in path.parts[-1]]
                 

    healthy_h5_paths = [path \
                 for path in list(Path(dataset_path).rglob("*random*.h5"))\
                 if fnmatch(path.parts[-1], '*empty*')
                 and not 'stroke' in path.parts[-1]]
    loc_label_path = dataset_path / 'location_class_random.h5'
else:
    total_h5_paths = [path \
                 for path in list(Path(dataset_path).rglob("*fixed*.h5"))\
                 if fnmatch(path.parts[-1], '*stroke*')
                 and not 'empty' in path.parts[-1]]
                 

    healthy_h5_paths = [path \
                 for path in list(Path(dataset_path).rglob("*fixed*.h5"))\
                 if fnmatch(path.parts[-1], '*empty*')
                 and not 'stroke' in path.parts[-1]]
    loc_label_path = dataset_path / 'location_class_fixed.h5'
# Shuffle and split file paths
shuffle_seed = 1253
random.seed(shuffle_seed)
random.shuffle(total_h5_paths)
random.seed(shuffle_seed)
random.shuffle(healthy_h5_paths)
#%%
split_ratio = 0.7
num_train= int(len(total_h5_paths) * split_ratio)


train_total_paths = total_h5_paths[:num_train]
train_healthy_paths = healthy_h5_paths[:num_train]

test_total_paths = total_h5_paths[num_train:]
test_healthy_paths = healthy_h5_paths[num_train:]


train_dataset = random_head_diffusion_loader(train_total_paths, train_healthy_paths, loc_label_path)
test_dataset = random_head_diffusion_loader(test_total_paths, test_healthy_paths, loc_label_path,
                                            train_dataset.norm_values_tt, train_dataset.norm_values_tr)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True)
#%%

for batch in train_loader:
    break

#%%
num_classes = 10
model = Unet(
    dim = 64,
    dim_mults = (1, 2, 4, 8),
    num_classes = num_classes,
    cond_drop_prob = 0.5
).cuda()

diffusion = GaussianDiffusion(
    model,
    image_size = 32,
    timesteps = 500    # number of steps
).cuda()

#%%
epochs = 180  # Define the number of epochs
optimizer = optim.Adam(model.parameters(), lr=8e-5)
epoch_loss_list = []  # Store average loss per epoch
for epoch in range(epochs):
    epoch_loss = 0.0
    for batch_idx, (images, image_classes) in enumerate(train_loader):
        images = images.cuda()
        image_classes = image_classes.cuda()
        
        optimizer.zero_grad()
        loss = diffusion(images, classes = image_classes)  # Forward pass and loss computation
        loss.backward()  # Backward pass
        optimizer.step()  # Update model parameters
        epoch_loss += loss.item()
        if batch_idx % 250 == 0:  # Log every 100 batches
            print(f'Epoch: {epoch}, Batch: {batch_idx}, Loss: {loss.item()}')
    
    # Calculate the average loss for the epoch
    avg_epoch_loss = epoch_loss / len(train_loader)
    epoch_loss_list.append(avg_epoch_loss)
    print(f'Epoch: {epoch}, Average Loss: {avg_epoch_loss}')
    print('='*100)
    # Here you would save or display sampled_images, but remember they're on CUDA and need processing
    if epoch % 10 == 0:
        # Simple timing for one sampling operation - place this where you need it
        start_time = time.time()
        class_to_sample = 5
        sample_classes = torch.tensor([class_to_sample for i in range(16)]).cuda()
        sampled_images = diffusion.sample(
            classes = sample_classes,
            cond_scale = 6.                # condition scaling, anything greater than 1 strengthens the classifier free guidance. reportedly 3-8 is good empirically
        )
        end_time = time.time()  # Capture the end time after the function call
        # Calculate the duration
        duration = end_time - start_time
        print(f"Sampling 1000 steps took {duration} seconds.")
        show_images(sampled_images, epoch+1)
        final_weights_path = root_path / f'ckpts/test'
        if not os.path.exists(final_weights_path):
            os.makedirs(final_weights_path)
        # if loss_G < best_loss:
            # lr_not_update_count = 0
            # best_loss = loss_G
        torch.save(diffusion.state_dict(), final_weights_path/f'diffusion{model_techs_sub}.pth')
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

