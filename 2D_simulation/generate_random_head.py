# -*- coding: utf-8 -*-
"""
Created on Wed Feb 28 10:28:31 2024

@author: User
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
#%% membrane + skin-skull + grey + white
import numpy as np
import cv2
import matplotlib.pyplot as plt

def generate_random_shift(max_shift):
    # Generate a random shift within the bounds
    shift_x = np.random.uniform(-max_shift, max_shift)
    shift_y = np.random.uniform(-max_shift, max_shift)
    return shift_x, shift_y

def make_grid(image, shape_1_size=16, shape_0_size=16):
    
    '''
    Objective: create grids for the frames
    input:
        first_frame: the frame of a video
        X_BLOCK_SIZE: length of x edge
        Y_BLOCK_SIZE: length of y edge

    Output:
        an 3 by n (total number of the grids) array. 
        x, y are the coordinate of the point of the top left corner
        [[x coordinate: ...],
         [y coordinate: ...],
         [grid number: ...]]
    
    '''
    
    height, width = image.shape[0], image.shape[1]
    total_pixel_of_grid = shape_1_size * shape_0_size
    # for storing x, y, grid index;
    grids = [[],[],[]]
    num_grids = 0
    # divide the x,y axis by the grid size respectively
    for y in range(int((height) / shape_0_size)):
        for x in range(int((width)/ shape_1_size)):

            current_x_pos = x*shape_1_size
            current_y_pos = y*shape_0_size
            # divide the grid using the index of the image
            current_grid = image[current_y_pos: current_y_pos+shape_0_size,
                                 current_x_pos: current_x_pos+shape_1_size]
            gird_in_ellipse_ratio = sum(sum((current_grid==255)*1)) / total_pixel_of_grid
            # if 80% of the grid is in the ssb area, then we count it
            if gird_in_ellipse_ratio>=0.8:
                grids[0].append(x*shape_1_size) 
                grids[1].append(y*shape_0_size)
                grids[2].append(num_grids)
                num_grids +=1
    return np.array(grids)

def plot_full_grids(total_grids):
    '''
    Objective:
        Just for plotting the grids inside the oval ssb and put order for each 
        grid using red text, and test the logic, not necessary for later usage.
    '''
    num_of_grids = total_grids.shape[1]
    grid_mask = np.zeros_like(boundary)
    color_grid = 255
    for g in range(num_of_grids):
        grid_mask = cv2.line(grid_mask, 
                         (total_grids[0][g]+shape_1_size, total_grids[1][g]),
                         (total_grids[0][g], total_grids[1][g]), color_grid, 1)
        grid_mask = cv2.line(grid_mask, 
                         (total_grids[0][g], total_grids[1][g]+shape_0_size),
                         (total_grids[0][g], total_grids[1][g]), color_grid, 1)
        
        grid_mask = cv2.line(grid_mask, 
                         (total_grids[0][g]+shape_1_size, total_grids[1][g]+shape_0_size),
                         (total_grids[0][g]+shape_1_size, total_grids[1][g]), color_grid, 1)
        grid_mask = cv2.line(grid_mask, 
                         (total_grids[0][g]+shape_1_size, total_grids[1][g]+shape_0_size),
                         (total_grids[0][g], total_grids[1][g]+shape_0_size), color_grid, 1)
        
        text_x = int((total_grids[0][g]+total_grids[0][g]+shape_1_size)/2)-5 # -5 for positioning 
        text_y = int((total_grids[1][g]+total_grids[1][g]+shape_0_size)/2)+3 # +3 for positioning                 
        plt.text(text_x, text_y, str(g), color="red", fontsize=6)
        
    plt.imshow(grid_mask)
    plt.show() 
    
def create_class_grids(image, total_grids, shape_1_size, shape_0_size):
    '''
    Objective:
        turn the image with stroke into array [1,n], n=number of grids
    Input: 
        image:the image plotted circle stroke using cv2
        total_grids: output from "make_grid", array [3,n]
        shape_1_size and shape_0_size for calculating the ratio
    Output:
        array [1,n], with n scores for each grid
    
    '''
    num_of_grids = total_grids.shape[1]
    # Since we use the portion of grids that is in stroke, instead of treating 
    # the entire stroke as a probability space that sum to 1. 
    total_pixel_of_grid = shape_1_size * shape_0_size
    # total_pixel_of_stroke = sum(sum((image==255)*1)) # for testing Alex method
    class_label = np.zeros([num_of_grids])
    for g in range(num_of_grids):

            current_x_pos = total_grids[0][g]
            current_y_pos = total_grids[1][g]
            current_grid = image[current_y_pos: current_y_pos+shape_0_size, current_x_pos: current_x_pos+shape_1_size]
            # calculate the ratio stroke/grid as the score for this grid
            # ==255 because color for cv2.cirle = 255
            gird_in_stroke_ratio = sum(sum((current_grid==255)*1)) / total_pixel_of_grid
            # gird_occupied_stroke_ratio = sum(sum((current_grid==255)*1)) / total_pixel_of_stroke 
            class_label[g] = np.round(gird_in_stroke_ratio,2) 
    
    return class_label


def plot_class_grids(total_grids, class_label, boundary, shape_1_size, shape_0_size,
                     if_save = False, exp_name = None, test_model=''):
    '''
    Objective:
        For visualizing the [1,n] class_label, or class_pred
        Plot in the simplest way (no color or boundary)
        Can be used to:
            1. plot label and prediction
            2. for 
    Input:
        total_grids: the [3,n] array
        class_label: the [1,n] array with score of each grid; output from "create_class_grids"        
    Output:
        grid_mask: just for testing, not necessary
    
    Note:
        The logic is same as the one in test_model, just the saving path differs
    
    '''
    
    num_of_grids = total_grids.shape[1]
    grid_mask = np.zeros_like(boundary)
    
    color_grid = 255
    for g in range(num_of_grids):
        
        if class_label[g] > 0:
 
            contours = np.array([[total_grids[0][g], total_grids[1][g]], [total_grids[0][g]+shape_0_size, total_grids[1][g]], \
                                 [total_grids[0][g]+shape_0_size, total_grids[1][g]+shape_1_size], [total_grids[0][g], total_grids[1][g]+shape_1_size]])  
            cv2.fillPoly(grid_mask, pts =[contours], color=int(color_grid*class_label[g]))
            text_x = int((total_grids[0][g]+total_grids[0][g]+shape_1_size)/2)-5 # -5 for positioning 
            text_y = int((total_grids[1][g]+total_grids[1][g]+shape_0_size)/2)+3 # +3 for positioning                 
            plt.text(text_x, text_y, str(g), color="red", fontsize=6)
            
    if if_save:   
        save_dir = root_path.parents[1]/f'Plots/{test_model}'
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        fname = f'{save_dir}/{exp_name}.jpg'
        plt.imshow(grid_mask)
        plt.title(f'{exp_name}')
        plt.savefig(fname, bbox_inches='tight', pad_inches=0, dpi=500)
        plt.close()
        
        
    else:
        plt.imshow(grid_mask)
        # plt.show() 
        plt.close() 
        return grid_mask


def plot_bound_stroke(boundary, shape_1_size, shape_0_size, save_dir, total_grids=None,
                      class_label=None,stroke_location=None, mcmap = 'Reds_r',
                      if_save = True, exp_name = None, save_dir_name='',
                      act_func='softmax', if_text_on=False):
    '''
    Objective:
        Main objective is to add the black boundary and turn the ROI to white,
        like what we did in fusion task. Extended it so it can plot 3 things
        1. All-grids: stroke_location=None, class_label=None
        2. Red stroke location: stroke_location=stroke_location, class_label=None
        3. Red stroke location overlapped by class label grids:
            stroke_location=stroke_location, class_label=class_label
        Can be optimized, but work well so keep it this for now.
    Input:
        boundary: SSB boundary
        total_grids: array [3, n]
        class_label: array [1, n]
        stroke_location: the circle stroke image
        save_dir_name: this is only used for saving at testing stage 
    Output:
        it will save the 3 aforementioned plots
    
    '''
    # plot the red stroke 
    if stroke_location is not None:
        sdata = np.copy(stroke_location).astype('float32')
        smin = np.nanmin(sdata)
        smax = np.nanmax(sdata)
        smed = (smax-smin)/2 + smin
        sdataX = np.copy(sdata)
        sdataX[sdata == 0] = np.NaN
        sdataX[sdataX < (6*(smax-smin)/10) + smin] = np.NaN
        plt.imshow(sdataX, cmap=mcmap, vmin=smed, vmax=smax*1.5)
    # plot the all-grid map
    else:
        if total_grids is not None and class_label is None:
            num_of_grids = total_grids.shape[1]
            grid_mask = np.zeros_like(boundary).astype('float32')
            grid_mask[grid_mask == 0] = np.NaN

            color_grid = 255
            # this loop is to plot lines for each grid, and put text on 
            for g in range(num_of_grids):
                grid_mask = cv2.line(grid_mask, 
                                 (total_grids[0][g]+shape_1_size, total_grids[1][g]),
                                 (total_grids[0][g], total_grids[1][g]), color_grid, 2) # the last int is the width of line 
                grid_mask = cv2.line(grid_mask, 
                                 (total_grids[0][g], total_grids[1][g]+shape_0_size),
                                 (total_grids[0][g], total_grids[1][g]), color_grid, 2)
                
                grid_mask = cv2.line(grid_mask, 
                                 (total_grids[0][g]+shape_1_size, total_grids[1][g]+shape_0_size),
                                 (total_grids[0][g]+shape_1_size, total_grids[1][g]), color_grid, 2)
                grid_mask = cv2.line(grid_mask, 
                                 (total_grids[0][g]+shape_1_size, total_grids[1][g]+shape_0_size),
                                 (total_grids[0][g], total_grids[1][g]+shape_0_size), color_grid, 2)
                contours = np.array([[total_grids[0][g], total_grids[1][g]], [total_grids[0][g]+shape_0_size, total_grids[1][g]], \
                                      [total_grids[0][g]+shape_0_size, total_grids[1][g]+shape_1_size], [total_grids[0][g], total_grids[1][g]+shape_1_size]])  
                cv2.fillPoly(grid_mask, pts =[contours], color=int(150))
                text_x = int((total_grids[0][g]+total_grids[0][g]+shape_1_size)/2)-5 # -5 for positioning 
                text_y = int((total_grids[1][g]+total_grids[1][g]+shape_0_size)/2)+3 # +3 for positioning                 
                # plt.text(text_x, text_y, str(g), color="black", fontsize=6)
                            
          
            plt.imshow(grid_mask, cmap='Blues_r', alpha=0.15)
            
    # plot the label region 
    if class_label is not None:
        reverse_val = class_label.max()
        num_of_grids = total_grids.shape[1]
        grid_mask = np.zeros_like(boundary)
        # threshold_array = class_label[class_label>0]
        # plot_threshold = np.percentile(threshold_array, 50)
        if act_func == 'softmax':
            plot_threshold=0.0
        else:
            plot_threshold=0.0
        color_grid = 255
        # this loop is for plotting the class grids
        for g in range(num_of_grids):
            # NNs output some small values, if the scores too low, just remove
            if class_label[g] > plot_threshold:
     
                contours = np.array([[total_grids[0][g], total_grids[1][g]], [total_grids[0][g]+shape_0_size, total_grids[1][g]], \
                                     [total_grids[0][g]+shape_0_size, total_grids[1][g]+shape_1_size], [total_grids[0][g], total_grids[1][g]+shape_1_size]])  
                # this line chaged to 1-class_label to reverse the grid color
                # cv2.fillPoly(grid_mask, pts =[contours], color=int(color_grid*(reverse_val+0.05-class_label[g])))
                cv2.fillPoly(grid_mask, pts =[contours], color=int(color_grid*(class_label[g])))

                if if_text_on:
                    text_x = int((total_grids[0][g]+total_grids[0][g]+shape_1_size)/2)-5 # -5 for positioning 
                    text_y = int((total_grids[1][g]+total_grids[1][g]+shape_0_size)/2)+3 # +3 for positioning                 
                    plt.text(text_x, text_y, str(g), color="black", fontsize=3)
        # for change the grid color to blue       
        grid_mask = grid_mask.astype('float32')
        smax = min(np.nanmax(grid_mask), 0.15*color_grid)
        if np.nanmax(class_label) < 0.075: # emperically defined 
            s_scale_factor = 2.1
        else:
            s_scale_factor = 1.1
        smin = np.nanmin(grid_mask)
        # when showing the grids turn off grids with too low value, but text remain        grid_mask[grid_mask < (1.5*(smax-smin)/10) + smin] = 0
        grid_mask[grid_mask == 0] = np.NaN
        # plt.imshow(grid_mask, cmap='Blues_r', vmax=smax*1.5, alpha=0.35)
        plt.imshow(grid_mask, cmap='Blues', alpha=0.55)

    else:
        pass
    
    #this part gives black boundary and white ROI
    sdataMask = np.copy(boundary).astype('float32')
    sdataMask[boundary > 0] = np.NaN
    plt.imshow(sdataMask, cmap='Greys_r')#, vmin=smed, vmax=smax*1.5)
    plt.axis('off')
    if if_save:   
        # save_dir = root_path.parents[1]/f'Score_plots/{save_dir_name}'
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        fname = f'{save_dir}/{exp_name}.jpg'

        # plt.title(f'{plot_name}_{exp_name}')
        plt.savefig(fname,bbox_inches='tight', pad_inches=0., dpi=500)
        plt.close()
    else:
        plt.show()
        
def write_to_h5(stroke_per_path, stroke_con_path, empty_per_path, empty_con_path,
                mask_per, mask_con, empty_mask_per, empty_mask_con,
                location_class_path, location_class):
    
    with h5py.File(stroke_per_path, 'a')  as stroke_per_h5, h5py.File(stroke_con_path, 'a')  as stroke_con_h5,\
        h5py.File(empty_per_path, 'a') as empty_per_h5, h5py.File(empty_con_path, 'a') as empty_con_h5,\
        h5py.File(location_class_path, 'a') as location_class_h5:
    
        stroke_per_h5.create_dataset(exp_id, data=mask_per.astype(np.float64))
        stroke_con_h5.create_dataset(exp_id, data=mask_con.astype(np.float64))
        empty_per_h5.create_dataset(exp_id, data=empty_mask_per.astype(np.float64))
        empty_con_h5.create_dataset(exp_id, data=empty_mask_con.astype(np.float64))
        location_class_h5.create_dataset(exp_id, data=location_class)

    
def create_layered_shape_with_fixed_ellipse(**kwargs):

    shape = kwargs['shape']
    center = kwargs['center']  
    base_axes = kwargs['base_axes']

    filled_liquid_di = kwargs['filled_liquid_di']
    scalp_di = kwargs['scalp_di']
    skull_di = kwargs['skull_di']
    grey_matter_di = kwargs['grey_matter_di']
    white_matter_di = kwargs['white_matter_di']

    fixed_ellipse_axes = kwargs['fixed_ellipse_axes']
    brain_ellipse_axes = kwargs['brain_ellipse_axes']
    bottom_amplitude = kwargs['bottom_amplitude']
    bottom_frequency = kwargs['bottom_frequency']
    top_amplitude = kwargs['top_amplitude']
    top_frequency = kwargs['top_frequency']
    num_layers = kwargs['num_layers']
    max_shift = kwargs['max_shift']
    skull_thickness = kwargs['skull_thickness']
    skin_thickness = kwargs['skin_thickness']

    angle_range = kwargs['angle_range']
    brain_shift_factor = kwargs['brain_shift_factor']
    stroke_axes = kwargs['stroke_axes']
    stroke_angle_range = kwargs['stroke_angle_range']
    stroke_type = kwargs['stroke_type']
    def generate_points(axes, center, rotation_matrix):
        points = []
        for theta in np.arange(0, 2 * np.pi, 2 * np.pi / 360):
            if theta > np.pi:  # Right side
                perturbation_factor = 1 + top_amplitude * np.sin(top_frequency * (theta - np.pi))
            else:  # Left and top side
                perturbation_factor = 1 + bottom_amplitude * np.sin(bottom_frequency * theta)

            r1 = axes[0] * perturbation_factor
            r2 = axes[1] * (perturbation_factor ** (np.cos(theta) ** 2))  # Adjust for head-like shape

            x = r1 * np.cos(theta)
            y = r2 * np.sin(theta)
            rotated_point = np.dot(rotation_matrix, np.array([x, y]))
            final_point = (int(rotated_point[0] + center[0]), int(rotated_point[1] + center[1]))
            points.append(final_point)
        return np.array([points])

    mask_per = np.zeros(shape, dtype=np.float32)
    mask_con = np.zeros(shape, dtype=np.float32)
    mask_for_fill = np.zeros_like(mask_con)
    # Draw fixed outer ellipse
    cv2.ellipse(mask_for_fill, center, fixed_ellipse_axes, 0, 0, 360, 2, -1)
    # print((mask_for_fill[mask_for_fill>0]==2).all())#anti-aliasing check
    # print((mask_for_fill[mask_for_fill!=2]).any())#anti-aliasing check

    mask_per[np.where(mask_for_fill!=0)] = filled_liquid_di[0]
    mask_con[np.where(mask_for_fill!=0)] = filled_liquid_di[1]
    boundary = mask_for_fill.copy()
    boundary[boundary!=0] = 255
    # cv2.ellipse(mask_per, center, fixed_ellipse_axes, 0, 0, 360, round(filled_liquid_di[0]*scale_factor), -1, lineType=cv2.LINE_AA)
    # cv2.ellipse(mask_con, center, fixed_ellipse_axes, 0, 0, 360, round(filled_liquid_di[1]*100), -1, lineType=cv2.LINE_AA)

    # Randomize the rotation and shift of the inner shape
    random_angle = np.random.uniform(*angle_range)
    rotation_matrix = np.array([[np.cos(np.deg2rad(random_angle)), -np.sin(np.deg2rad(random_angle))], [np.sin(np.deg2rad(random_angle)), np.cos(np.deg2rad(random_angle))]])
    shift_x, shift_y = generate_random_shift(max_shift)
    shifted_center = (int(center[0] + shift_x), int(center[1] + shift_y))
    # print(base_axes)
    # Create layers for the irregular shape
    layer_increment = 0
    permittivity_list = [scalp_di[0], skull_di[0] , grey_matter_di[0], white_matter_di[0]]
    conductivity_list = [scalp_di[1], skull_di[1] , grey_matter_di[1], white_matter_di[1]]

    for i in range(num_layers):
        if i == 1:
            layer_increment += skull_thickness
        else:
            layer_increment += skin_thickness
        layer_axes = (base_axes[0] - layer_increment, base_axes[1] - layer_increment)
        # print(layer_axes)
        mask_for_fill = np.zeros_like(mask_con)
        cv2.fillPoly(mask_for_fill, generate_points(layer_axes, shifted_center, rotation_matrix), color=2, lineType=cv2.LINE_AA)
        # print((mask_for_fill[mask_for_fill>0]==2).all())#anti-aliasing check
        mask_per[np.where(mask_for_fill!=0)] = permittivity_list[i]
        mask_con[np.where(mask_for_fill!=0)] = conductivity_list[i]
        # cv2.fillPoly(mask_per, generate_points(layer_axes, shifted_center, rotation_matrix), color=round(permittivity_list[i]*scale_factor), lineType=cv2.LINE_AA)
        # cv2.fillPoly(mask_con, generate_points(layer_axes, shifted_center, rotation_matrix), color=round(conductivity_list[i]*100), lineType=cv2.LINE_AA)

    # Draw the inner irregular brain-like ellipse within the inner part of the original ellipse

    brain_shifted_center = (int(center[0] + shift_x * brain_shift_factor[0]),
                            int(center[1] + shift_y * brain_shift_factor[1]))  # Slight shift for the brain ellipse
    # brain_shifted_center = shifted_center
    mask_for_fill = np.zeros_like(mask_con)
    cv2.fillPoly(mask_for_fill, generate_points(brain_ellipse_axes, brain_shifted_center, rotation_matrix), color=2, lineType=cv2.LINE_AA)
    # print((mask_for_fill[mask_for_fill>0]==2).all())#anti-aliasing check
    mask_per[np.where(mask_for_fill!=0)] = permittivity_list[-1]
    mask_con[np.where(mask_for_fill!=0)] = conductivity_list[-1]
    # cv2.fillPoly(mask_per, generate_points(brain_ellipse_axes, brain_shifted_center, rotation_matrix), color=round(permittivity_list[-1]*scale_factor), lineType=cv2.LINE_AA)
    # cv2.fillPoly(mask_con, generate_points(brain_ellipse_axes, brain_shifted_center, rotation_matrix), color=round(conductivity_list[-1]*100), lineType=cv2.LINE_AA)

    empty_mask_per = mask_per.copy() 
    empty_mask_con = mask_con.copy() 

    # insert random stroke
    insert_stroke_region = np.zeros_like(mask_per)
    
    '''
    comment below if no stroke 
    '''
    stroke_regio_axes = np.array(brain_ellipse_axes)-20
    brain_fill = 1
    temp_fill = 5
    cv2.fillPoly(insert_stroke_region, generate_points(brain_ellipse_axes, brain_shifted_center, rotation_matrix), color=brain_fill, lineType=cv2.LINE_AA)
    brain_region = insert_stroke_region.copy()
    cv2.fillPoly(insert_stroke_region, generate_points(stroke_regio_axes, brain_shifted_center, rotation_matrix), color=temp_fill, lineType=cv2.LINE_AA)
    random_idx= np.random.randint(0, np.array(np.where(insert_stroke_region==temp_fill)).shape[1]-1)

    random_stroke_center =  tuple(np.array(np.where(insert_stroke_region==temp_fill))[:, random_idx])
    random_angle = np.random.uniform(*stroke_angle_range)
    if stroke_type == 0:
        stroke_fill_per = white_matter_di[0]*0.8 # 80% of normal tissue
        stroke_fill_con = white_matter_di[1]*0.8 # 80% of normal tissue
    elif stroke_type == 1: 
        stroke_fill_per = 61.1 # blood per at 1GHz
        stroke_fill_con = 1.58 # blood con at 1GHz
    cv2.ellipse(insert_stroke_region, (random_stroke_center),stroke_axes , random_angle, 0, 360, 2,-1, lineType=cv2.LINE_AA)
    insert_stroke_region[insert_stroke_region!= 2] = 0
    # print((insert_stroke_region[insert_stroke_region>0]==2).all())#anti-aliasing check
    insert_stroke_region *= brain_region
    mask_per[np.where(insert_stroke_region !=0)] = stroke_fill_per
    mask_con[np.where(insert_stroke_region !=0)] = stroke_fill_con
    # mask_per = mask_per.astype(np.float32)
    # mask_con = mask_con.astype(np.float32)
    stroke_location = insert_stroke_region.copy()
    stroke_location[stroke_location==2] = 255
    
    mask_per[boundary==0] = filled_liquid_di[0]
    mask_con[boundary==0] = filled_liquid_di[1]
    empty_mask_per[boundary==0] = filled_liquid_di[0]
    empty_mask_con[boundary==0] = filled_liquid_di[1]
    mask_per = cv2.resize(mask_per, (256,256))
    mask_con = cv2.resize(mask_con, (256,256))
    empty_mask_per = cv2.resize(empty_mask_per, (256,256))
    empty_mask_con = cv2.resize(empty_mask_con, (256,256))

    boundary = cv2.resize(boundary, (256,256))
    stroke_location = cv2.resize(stroke_location, (256,256))
    return mask_per, mask_con, empty_mask_per, empty_mask_con, boundary, stroke_location

if __name__ == '__main__':
    if_random = False
    if not if_random:
        fixed = '_fixed' # ['', '_fixed']
    else:
        fixed = '_random'
    
    brain_gap_factor = 1 # this factor set to 1 means no grey between white matter and skull
    brain_shift_factor = 1 # set to 1 means brain no shift within skull
    shape_1_size, shape_0_size = 8, 8
    # Step 1: Define independent parameters
    fixed_parameters = {
        'num_layers': 2,
        'max_shift': [4 if if_random else 0 for i in range(1)][0],  # Maximum shift within the fixed ellipse
        'angle_range': [(-3,3) if if_random else (0,0) for i in range(1)][0],
        'shape': (512, 512),
        'center': (256, 256),
        'fixed_ellipse_axes': (90, 105),
        'filled_liquid_di': (40, 0.1),
        'scalp_di': (40.9,0.9),
        'skull_di': (12.9, 0.16),
        'grey_matter_di': (52.3,0.98),
        'white_matter_di': (38.6, 0.6),
        'frequency_arr': [1, 3], # frequency for top and bottom boundary
    
        'skull_thickness': np.random.uniform(7, 8) / 2,
        'skin_thickness': np.random.uniform(5, 7) / 2,
        'head_width': np.random.uniform(142, 151) / 2,
        'head_length': np.random.uniform(182, 200) / 2,
    }
    
    # Update fixed_parameters with dependent parameters
    fixed_parameters['ss_width'] = fixed_parameters['skull_thickness'] + fixed_parameters['skin_thickness'] * 2
    fixed_parameters['base_axes'] = (fixed_parameters['head_width'], fixed_parameters['head_length'])
    fixed_parameters['brain_ellipse_axes'] = ((fixed_parameters['head_width'] - fixed_parameters['ss_width']) * brain_gap_factor,
                                              (fixed_parameters['head_length'] - fixed_parameters['ss_width']) * brain_gap_factor)
    fixed_parameters['brain_shift_factor'] = [np.random.uniform(brain_shift_factor, 1), np.random.uniform(brain_shift_factor, 1)]
    fixed_parameters['bottom_frequency'] = 3 #np.random.choice(fixed_parameters['frequency_arr'], p=[0.3, 0.7])
    fixed_parameters['bottom_amplitude'] = 0.05 #np.random.uniform(0.05, 0.15)
    fixed_parameters['top_frequency'] = 1 #np.random.choice(fixed_parameters['frequency_arr'], p=[0.7, 0.3])
    fixed_parameters['top_amplitude'] = 0.1# np.random.uniform(0.05, 0.15)
    # Ensure top_amplitude is within the bounds after being randomly chosen
    if fixed_parameters['top_frequency'] == 3:
        fixed_parameters['top_amplitude'] = np.clip(fixed_parameters['top_amplitude'], 0.01, 0.05)
    
    # Generate the mask
    num_cases = 20000
    case_already_have = 20000
    num_in_partition = 2000
    # write to h5

    
    for i in trange(case_already_have, case_already_have+num_cases):
        np.random.seed(i)  # Set seed for reproducibility
        partition = i // num_in_partition + 1

        stroke_per_h5 = root_path / f'data/stroke_per{fixed}_{partition}.h5'
        # if Path.is_file(stroke_per_h5):
        #     os.remove(stroke_per_h5)   
        stroke_con_h5 = root_path / f'data/stroke_con{fixed}_{partition}.h5'     
        # if Path.is_file(stroke_con_h5):
        #     os.remove(stroke_con_h5)   
        empty_per_h5 = root_path / f'data/empty_per{fixed}_{partition}.h5'
        # if Path.is_file(empty_per_h5):
        #     os.remove(empty_per_h5) 
        empty_con_h5 = root_path / f'data/empty_con{fixed}_{partition}.h5'
        # if Path.is_file(empty_con_h5):
        #     os.remove(empty_con_h5) 
        location_class_h5 = root_path / f'data/location_class{fixed}.h5' # need to increment this one
        
        if if_random:
            fixed_parameters['skull_thickness'] = np.random.uniform(7, 8) / 2
            fixed_parameters['skin_thickness'] = np.random.uniform(5, 7) / 2
            fixed_parameters['head_width'] = np.random.uniform(142, 151) / 2
            fixed_parameters['head_length'] = np.random.uniform(182, 200) / 2
            # Update fixed_parameters with dependent parameters
            fixed_parameters['ss_width'] = fixed_parameters['skull_thickness'] + fixed_parameters['skin_thickness'] * 2
            fixed_parameters['base_axes'] = (fixed_parameters['head_width'], fixed_parameters['head_length'])
            fixed_parameters['fixed_ellipse_axes'] = (90, 105)
            fixed_parameters['brain_ellipse_axes'] = ((fixed_parameters['head_width'] - fixed_parameters['ss_width']) * brain_gap_factor,
                                                      (fixed_parameters['head_length'] - fixed_parameters['ss_width']) * brain_gap_factor)
            fixed_parameters['brain_shift_factor'] = [np.random.uniform(brain_shift_factor, 1), np.random.uniform(brain_shift_factor, 1)]
            # fixed_parameters[] = 
            fixed_parameters['bottom_frequency'] = np.random.choice(fixed_parameters['frequency_arr'], p=[0.3, 0.7])
            fixed_parameters['bottom_amplitude'] = np.random.uniform(0.05, 0.15)
            fixed_parameters['top_frequency'] = np.random.choice(fixed_parameters['frequency_arr'], p=[0.7, 0.3])
            fixed_parameters['top_amplitude'] = np.random.uniform(0.05, 0.15)
            # Ensure top_amplitude is within the bounds after being randomly chosen
            if fixed_parameters['top_frequency'] == 3:
                fixed_parameters['top_amplitude'] = np.clip(fixed_parameters['top_amplitude'], 0.01, 0.05)
                
        #stroke parameters
        stroke_parameters = {
            'stroke_axes' : (np.random.randint(5, 18) , np.random.randint(5, 18)), 
            'stroke_angle_range' : (0, 360), 
            'stroke_type' : np.random.choice([0, 1], 1, p=[0.5, 0.5]), 
        }
        if stroke_parameters['stroke_type'] ==0:
            stroke_type = 'ISC'
        elif stroke_parameters['stroke_type'] ==1:
            stroke_type = 'HAE'
        exp_id = f'Exp{str(i).zfill(5)}_{stroke_type}{fixed}-head-v0'
        exp_id_empty = f'Exp{str(i).zfill(5)}_empty{fixed}-head-v0'
    
        # mask_per, mask_con, insert_stroke_region = create_layered_shape_with_fixed_ellipse(shape, center, base_axes,
        #                                                fixed_ellipse_axes,
        #                                                brain_ellipse_axes,
        #                                                bottom_amplitude, bottom_frequency,
        #                                                top_amplitude, top_frequency,
        #                                                num_layers, max_shift, angle_range,
        #                                                brain_shift_factor,
        #                                                stroke_axes, stroke_angle_range, stroke_type)
        mask_per, mask_con, empty_mask_per, empty_mask_con, boundary, stroke_location\
            = create_layered_shape_with_fixed_ellipse(**fixed_parameters, **stroke_parameters)
        
        if not if_random: 
            assert fixed_parameters['top_frequency'] == 1
            assert fixed_parameters['bottom_frequency'] == 3
        #%% location class part
        total_grids = make_grid(boundary, shape_1_size, shape_0_size)
        # plot_full_grids(total_grids)
        # note this class it grid location class
        class_label = create_class_grids(stroke_location, total_grids, shape_1_size, shape_0_size)
        class_label = class_label/class_label.sum()
        # print(f'class_label 1st: {class_label.sum()}')
        # grid_mask = plot_class_grids(total_grids, class_label, boundary, shape_1_size, shape_0_size)
        # print(f'class_label after plot_class_grids: {class_label.sum()}')
    
    
            
        write_to_h5(stroke_per_h5, stroke_con_h5, empty_per_h5, empty_con_h5,
                        mask_per, mask_con, empty_mask_per, empty_mask_con,
                        location_class_h5, class_label)
        # save per
        plt.imshow(mask_per, cmap='jet')
        plt.colorbar()
        plt.title(f'{stroke_type} brain permittivity')
        plots_dir = './data/plots/permittivity'
        if not os.path.exists(plots_dir):
            os.makedirs(plots_dir)
        plt.tight_layout()
        plt.savefig(f'{plots_dir}/{exp_id}_per.png')
        plt.close()
        # save stroke location class
        plot_bound_stroke(boundary, shape_1_size, shape_0_size, plots_dir, total_grids,
                          class_label=class_label, stroke_location = stroke_location,
                          if_save=True, exp_name=exp_id+'_stroke_label'
                          )
        # save con
        plt.imshow(mask_con, cmap='jet')
        plt.colorbar()
        plt.title(f'{stroke_type} brain conductivity')
        plots_dir = './data/plots/conductivity'
        if not os.path.exists(plots_dir):
            os.makedirs(plots_dir)
        plt.tight_layout()
        plt.savefig(f'{plots_dir}/{exp_id}_con.png')
        plt.close()
        
        # save empty per
        plt.imshow(empty_mask_per, cmap='jet')
        plt.colorbar()
        plt.title('empty brain permittivity')
        plots_dir = './data/plots/permittivity_empty'
        if not os.path.exists(plots_dir):
            os.makedirs(plots_dir)
        plt.tight_layout()
        plt.savefig(f'{plots_dir}/{exp_id}_emptyPer.png')
        plt.close()
        
        # save empty con
        plt.imshow(empty_mask_con, cmap='jet')
        plt.colorbar()
        plt.title('empty brain conductivity')
        plots_dir = './data/plots/conductivity_empty'
        if not os.path.exists(plots_dir):
            os.makedirs(plots_dir)
        plt.tight_layout()
        plt.savefig(f'{plots_dir}/{exp_id}_emptyCon.png')
        plt.close()


# per_dict = {'random_eps': np.transpose(np.array(mask_per_list, dtype=np.float64),(1,2,0))}
# savemat(r'./data/random_eps.mat', per_dict)

# con_dict = {'random_sigma': np.transpose(np.array(mask_con_list, dtype=np.float64),(1,2,0))}
# savemat(r'./data/random_sigma.mat', con_dict)


# print(f'stroke type is: {stroke_parameters["stroke_type"]}, max per = {mask_per.max()}')
# np.save('./data/random_eps.npy', np.transpose(np.array(mask_per_list),(2,1,0)))
# np.save('./data/random_sigma.npy', np.transpose(np.array(mask_con_list),(2,1,0)))

#%% **kwargs example

'''
 # head parameters
num_layers = 3
max_shift = 4  # Maximum shift within the fixed ellipse

filled_liquid_per = 45
grey_matter_per = 52.3
scalp_per = 40.9
skull_per = 12.9
white_matter_per = 38.6
skull_thickness = np.random.uniform(7, 8, 1)/2
skin_thickness = np.random.uniform(5, 7, 1)/2
head_width = np.random.uniform(142, 151, 1)/2
head_length = np.random.uniform(182, 200, 1)/2
ss_width = (skull_thickness + skin_thickness*2)
# outline Parameters
shape = (256, 256)
center = (128, 128)
# brain parameters
base_axes = (head_width , head_length)  # Innermost irregular shape axes
fixed_ellipse_axes = (90, 105)  # Fixed outer ellipse size
brain_ellipse_axes = ((head_width-ss_width)*0.95, (head_length-ss_width)*0.95)  # Inner brain-like irregular ellipse size
brain_shift_factor = [np.random.uniform(0.85, 0.1, 1), np.random.uniform(0.85, 1, 1)]
angle_range = (0, 3)  # Allowable rotation range for the irregular shape, deg

frequency_arr = [1, 3]
bottom_frequency = np.random.choice(frequency_arr, 1, p=[0.3, 0.7])
# bottom_frequency = 3
bottom_amplitude = np.random.uniform(0.05, 0.15, 1) # if 1 between 0.01~0.15, default = 0.1
# bottom_amplitude = 0.15
# 1 or 3
top_frequency = np.random.choice(frequency_arr, 1, p=[0.7, 0.3])
top_amplitude = np.random.uniform(0.05, 0.015, 1) # if 3 <=0.05
if top_frequency == 3:
    top_amplitude = np.clip(top_amplitude, 0.01, 0.05)
'''
'''
def create_layered_shape_with_fixed_ellipse(**kwargs):
     # Your function implementation, accessing parameters via kwargs
    # kwargs.get('parameter_name', 'default_value')
    print(kwargs['stroke_type'])

      # Implement your function here
# Define fixed parameters outside the loop
fixed_parameters = {
    'shape': (256, 256),
    'center': (128, 128),
    # Define other fixed parameters
    'base_axes': (70, 85),  # Example fixed parameter
    'fixed_ellipse_axes': (90, 105),  # Another example
    # Add all other fixed parameters here
}
num_cases = 50
for i in range(num_cases):
    np.random.seed(i)  # Optional: Set seed for reproducibility of random variables

    # Update variable parameters for each iteration
    variable_parameters = {
        'stroke_type': np.random.choice([0, 1], p=[0.5, 0.5])  # This parameter changes every iteration
        # Add any other variable parameters here
    }   
    create_layered_shape_with_fixed_ellipse(shape, center, **variable_parameters, **fixed_parameters)
#%% membrane + skin-skull + grey
import numpy as np
import cv2
import matplotlib.pyplot as plt

def generate_random_shift(max_shift):
    # Generate a random shift within the bounds
    shift_x = np.random.uniform(-max_shift, max_shift)
    shift_y = np.random.uniform(-max_shift, max_shift)
    return shift_x, shift_y

def create_layered_shape_with_fixed_ellipse(shape, center, base_axes, fixed_ellipse_axes, amplitude, frequency, right_amplitude, right_frequency, num_layers, layer_increment, max_shift, angle_range):
    def generate_points(axes, center, angle):
        points = []
        for theta in np.arange(0, 2 * np.pi, 2 * np.pi / 360):
            if theta > np.pi:  # Right side
                perturbation_factor = 1 + right_amplitude * np.sin(right_frequency * (theta - np.pi))
            else:  # Left and top side
                perturbation_factor = 1 + amplitude * np.sin(frequency * theta)

            r1 = axes[0] * perturbation_factor
            r2 = axes[1] * (perturbation_factor ** (np.cos(theta) ** 2))  # Adjust for head-like shape

            x = r1 * np.cos(theta)
            y = r2 * np.sin(theta)
            rotated_point = np.dot(rotation_matrix, np.array([x, y]))
            final_point = (int(rotated_point[0] + center[0]), int(rotated_point[1] + center[1]))
            points.append(final_point)
        return np.array([points])

    mask = np.zeros(shape, dtype=np.uint8)

    # Draw fixed outer ellipse
    cv2.ellipse(mask, center, fixed_ellipse_axes, 0, 0, 360, num_layers + 2, -1)

    # Randomize the rotation and shift of the inner shape
    random_angle = np.random.uniform(*angle_range)
    rotation_matrix = np.array([[np.cos(np.deg2rad(random_angle)), -np.sin(np.deg2rad(random_angle))], [np.sin(np.deg2rad(random_angle)), np.cos(np.deg2rad(random_angle))]])
    shift_x, shift_y = generate_random_shift(max_shift)
    shifted_center = (int(center[0] + shift_x), int(center[1] + shift_y))
    
    print(base_axes)
    # Create layers for the irregular shape
    for i in range(num_layers, 0, -1):
        layer_axes = (base_axes[0] + i * layer_increment, base_axes[1] + i * layer_increment)
        print(layer_axes)
        cv2.fillPoly(mask, generate_points(layer_axes, shifted_center, random_angle), color=i+1)

    return mask

# Parameters
shape = (600, 800)
center = (400, 300)
base_axes = (100, 150)  # Innermost irregular shape axes
fixed_ellipse_axes = (160, 210)  # Fixed outer ellipse size
angle_range = (0, 0)  # Allowable rotation range for the irregular shape
amplitude = 0.015
frequency = 1.
right_amplitude = 0.05
right_frequency = 3
num_layers = 3
layer_increment = 5
max_shift = 0  # Maximum shift within the fixed ellipse

# Generate the mask
mask = create_layered_shape_with_fixed_ellipse(shape, center, base_axes, fixed_ellipse_axes, amplitude, frequency, right_amplitude, right_frequency, num_layers, layer_increment, max_shift, angle_range)

# Visualize the result
plt.imshow(mask, cmap='jet')
plt.colorbar()
plt.title('Fixed Ellipse with Random Inner Shapes')
plt.show()

#%% increase thincknees of outter layer (skin)

import numpy as np
import cv2
import matplotlib.pyplot as plt

def generate_random_shift(max_shift):
    shift_x = np.random.uniform(-max_shift, max_shift)
    shift_y = np.random.uniform(-max_shift, max_shift)
    return shift_x, shift_y

def create_custom_head_mask(shape, center, base_axes, fixed_ellipse_axes, amplitude, frequency, right_amplitude, right_frequency, num_layers, layer_increment, max_shift, angle_range, thickness_variation):
    def generate_points(axes, center, angle, apply_variation=False):
        points = []
        for theta in np.arange(0, 2 * np.pi, 2 * np.pi / 360):
            if theta > np.pi:  # Right side
                perturbation_factor = 1 + right_amplitude * np.sin(right_frequency * (theta - np.pi))
            else:  # Left and top side
                perturbation_factor = 1 + amplitude * np.sin(frequency * theta)

            if apply_variation:
                if (theta <= np.pi/2*1.3 and theta > np.pi/4*1.4):  # Apply thickness variation to the top and bottom
                    perturbation_factor += thickness_variation
                elif (theta <= 3*np.pi/2*1.1 and theta > 3*np.pi/2*0.9): 
                    perturbation_factor += thickness_variation
            r1 = axes[0] * perturbation_factor
            r2 = axes[1] * (perturbation_factor ** (np.cos(theta) ** 2))  # Adjust for head-like shape

            x = r1 * np.cos(theta)
            y = r2 * np.sin(theta)
            rotated_point = np.dot(rotation_matrix, np.array([x, y]))
            final_point = (int(rotated_point[0] + center[0]), int(rotated_point[1] + center[1]))
            points.append(final_point)
        return np.array([points])

    mask = np.zeros(shape, dtype=np.uint8)

    # Draw fixed outer ellipse
    cv2.ellipse(mask, center, fixed_ellipse_axes, 0, 0, 360, num_layers + 2, -1)

    # Randomize the rotation and shift
    random_angle = np.random.uniform(*angle_range)
    rotation_matrix = np.array([[np.cos(np.deg2rad(random_angle)), -np.sin(np.deg2rad(random_angle))], [np.sin(np.deg2rad(random_angle)), np.cos(np.deg2rad(random_angle))]])
    shift_x, shift_y = generate_random_shift(max_shift)
    shifted_center = (int(center[0] + shift_x), int(center[1] + shift_y))
    
    # Draw the layers
    for i in range(num_layers, 0, -1):
        layer_axes = (base_axes[0] + i * layer_increment, base_axes[1] + i * layer_increment)
        if i == 1:  # Apply thickness variation to the first layer
            cv2.fillPoly(mask, generate_points(layer_axes, shifted_center, random_angle, apply_variation=True), color=i+1)
        else:
            cv2.fillPoly(mask, generate_points(layer_axes, shifted_center, random_angle), color=i+1)

    return mask

# Parameters
shape = (600, 800)
center = (400, 300)
base_axes = (100, 150)
fixed_ellipse_axes = (200, 250)
angle_range = (0, 0)
amplitude = 0.015
frequency = 1.
right_amplitude = 0.05
right_frequency = 3
num_layers = 3
layer_increment = 5
max_shift = 0
thickness_variation = 0.08 # Control the variation in thickness

# Generate the mask
mask = create_custom_head_mask(shape, center, base_axes, fixed_ellipse_axes, amplitude, frequency, right_amplitude, right_frequency, num_layers, layer_increment, max_shift, angle_range, thickness_variation)

# Visualize the result
plt.imshow(mask, cmap='jet')
plt.colorbar()
plt.title('Custom Head Mask with Thickness Variation')
plt.show()

#%% smooth the thickness but skewed to oneside
import numpy as np
import cv2
import matplotlib.pyplot as plt

def smoothstep(edge0, edge1, x):
    # Scale, bias and saturate x to 0..1 range
    x = np.clip((x - edge0) / (edge1 - edge0), 0.0, 1.0)
    # Evaluate polynomial
    return x * x * (2.5 - 2 * x)

def generate_random_shift(max_shift):
    shift_x = np.random.uniform(-max_shift, max_shift)
    shift_y = np.random.uniform(-max_shift, max_shift)
    return shift_x, shift_y

def create_custom_head_mask(shape, center, base_axes, fixed_ellipse_axes, amplitude, frequency, right_amplitude, right_frequency, num_layers, layer_increment, max_shift, angle_range, thickness_variation):
    def generate_points(axes, center, angle, apply_variation=False):
        points = []
        for theta in np.arange(0, 2 * np.pi, 2 * np.pi / 360):
            if theta > np.pi:  # Right side
                perturbation_factor = 1 + right_amplitude * np.sin(right_frequency * (theta - np.pi))
            else:  # Left and top side
                perturbation_factor = 1 + amplitude * np.sin(frequency * theta)

            if apply_variation:
                # Define the regions for smooth transitions
                if (theta <= np.pi/2*1.3 and theta > np.pi/4*1.4):  # Transition to top
                    transition_factor = smoothstep(np.pi/4*1.4, np.pi/2*1.3, theta)
                    perturbation_factor += thickness_variation * transition_factor
                elif (theta <= 3*np.pi/2*1.1 and theta > 3*np.pi/2*0.9):  # Transition to bottom
                    transition_factor = smoothstep(3*np.pi/2*0.9, 3*np.pi/2*1.1, theta)
                    perturbation_factor += thickness_variation * transition_factor

            r1 = axes[0] * perturbation_factor
            r2 = axes[1] * perturbation_factor  # Simplified for smoother transitions

            x = r1 * np.cos(theta)
            y = r2 * np.sin(theta)
            rotated_point = np.dot(rotation_matrix, np.array([x, y]))
            final_point = (int(rotated_point[0] + center[0]), int(rotated_point[1] + center[1]))
            points.append(final_point)
        return np.array([points])

    mask = np.zeros(shape, dtype=np.uint8)
    random_angle = np.random.uniform(*angle_range)
    rotation_matrix = np.array([[np.cos(np.deg2rad(random_angle)), -np.sin(np.deg2rad(random_angle))], [np.sin(np.deg2rad(random_angle)), np.cos(np.deg2rad(random_angle))]])
    shift_x, shift_y = generate_random_shift(max_shift)
    shifted_center = (int(center[0] + shift_x), int(center[1] + shift_y))

    # Draw fixed outer ellipse
    cv2.ellipse(mask, center, fixed_ellipse_axes, 0, 0, 360, num_layers + 2, -1)

    # Draw the layers
    for i in range(num_layers, 0, -1):
        layer_axes = (base_axes[0] + i * layer_increment, base_axes[1] + i * layer_increment)
        if i == num_layers:  # Apply thickness variation to the first layer
            cv2.fillPoly(mask, generate_points(layer_axes, shifted_center, random_angle, apply_variation=True), color=i+1)
        else:
            cv2.fillPoly(mask, generate_points(layer_axes, shifted_center, random_angle), color=i+1)

    return mask

# Parameters
shape = (600, 800)
center = (400, 300)
base_axes = (100, 150)
fixed_ellipse_axes = (200, 250)
angle_range = (0, 0)
amplitude = 0.015
frequency = 1.
right_amplitude = 0.05
right_frequency = 3
num_layers = 3
layer_increment = 5
max_shift = 0
thickness_variation = 0.05 # Control the variation in thickness

# Generate the mask
mask = create_custom_head_mask(shape, center, base_axes, fixed_ellipse_axes, amplitude, frequency, right_amplitude, right_frequency, num_layers, layer_increment, max_shift, angle_range, thickness_variation)
# Visualize the result
plt.imshow(mask, cmap='jet')
plt.colorbar()
plt.title('Custom Head Mask with Thickness Variation')
plt.show()
'''