import torch
torch.set_printoptions(profile="full")

import numpy as np
import random
import argparse
import sys
from pathlib import Path
from core.cfgs import cfg, parse_args

def maskPointSelector(imgnum):
    masks_fore = np.load(f'./masks/masks_fore_im{imgnum}.npy', allow_pickle=True)
    mask_fore_x_list = masks_fore.item()['mask_fore_points_x']
    fore_label = [i for i in range(len(mask_fore_x_list))]
    fore_label = random.sample(fore_label,400)

    mask_points_fore_x = [mask_fore_x_list[i] for i in fore_label] # 400 sampling foreground points for one batch
    mask_points_fore_x = np.array(mask_points_fore_x)
    # print(mask_points_x)
    # print(mask_points.shape)
    x_coords = [i for i in mask_points_fore_x]
    x_coords_min = np.min(x_coords)
    x_coords_max = np.max(x_coords)
    # map coords to (-1,1)
    a = -1
    b = 1
    mask_fore_points_x_map = a + (b-a)/(x_coords_max-x_coords_min)*(mask_points_fore_x-x_coords_min)
    mask_fore_points_x_map = np.array(mask_fore_points_x_map, dtype=np.float64)

    mask_fore_y_list = masks_fore.item()['mask_fore_points_y']
    mask_points_fore_y = [mask_fore_y_list[i] for i in fore_label]
    mask_points_fore_y = np.array(mask_points_fore_y)
    # print(mask_points.shape)
    y_coords = [i for i in mask_points_fore_y]
    y_coords_min = np.min(y_coords)
    y_coords_max = np.max(y_coords)
    # map coords to (-1,1)
    a = -1
    b = 1
    mask_fore_points_y_map = a + (b-a)/(y_coords_max-y_coords_min)*(mask_points_fore_y-y_coords_min)
    mask_fore_points_y_map = np.array(mask_fore_points_y_map, dtype=np.float64)

    ## 
    masks_back = np.load(f'./data/masks/masks_back_im{imgnum}.npy', allow_pickle=True)
    mask_back_x_list = masks_back.item()['mask_back_points_x']
    back_label = [i for i in range(len(mask_back_x_list))]
    back_label = random.sample(back_label,41)

    mask_points_back_x = [mask_back_x_list[i] for i in back_label] # 41 sampling background points for one batch
    mask_points_back_x = np.array(mask_points_back_x)
    # print(mask_points_x)
    # print(mask_points.shape)
    x_coords = [i for i in mask_points_back_x]
    x_coords_min = np.min(x_coords)
    x_coords_max = np.max(x_coords)
    # map coords to (-1,1)
    a = -1
    b = 1
    mask_back_points_x_map = a + (b-a)/(x_coords_max-x_coords_min)*(mask_points_back_x-x_coords_min)
    mask_back_points_x_map = np.array(mask_back_points_x_map, dtype=np.float64)

    mask_back_y_list = masks_back.item()['mask_back_points_y']
    mask_points_back_y = [mask_back_y_list[i] for i in back_label]
    mask_points_back_y = np.array(mask_points_back_y)
    # print(mask_points.shape)
    y_coords = [i for i in mask_points_back_y]
    y_coords_min = np.min(y_coords)
    y_coords_max = np.max(y_coords)
    # map coords to (-1,1)
    a = -1
    b = 1
    mask_back_points_y_map = a + (b-a)/(y_coords_max-y_coords_min)*(mask_points_back_y-y_coords_min)
    mask_back_points_y_map = np.array(mask_back_points_y_map, dtype=np.float64)

    mask_points_x_map = np.concatenate((mask_fore_points_x_map,mask_back_points_x_map), axis=0)
    mask_points_y_map = np.concatenate((mask_fore_points_y_map,mask_back_points_y_map), axis=0)

    return mask_points_x_map, mask_points_y_map

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--imgrange', type=int, 
                    help="range of image name")
    parser.add_argument('--cfg_file', type=str, default='configs/pymaf_config.yaml',
                        help='config file path.')
    parser.add_argument('--misc', default=None, type=str, nargs="*",
                    help='other parameters')

    args = parser.parse_args()
    parse_args(args)

    mask_points_x = []
    mask_points_y = []
    for imgnum in range(1,args.imgrange+1):
        imgnum_fmt = "{:0>5d}".format(imgnum)
        try:
            mask_points_x_map, mask_points_y_map= maskPointSelector(imgnum_fmt)
            mask_points_x.append(mask_points_x_map)
            mask_points_y.append(mask_points_y_map)
        except:
            pass

    mask_points = torch.stack([torch.tensor(mask_points_x), torch.tensor(mask_points_y)]) # [2,61,441], the first 2 is the length of imgname args
    mask_points_grid = torch.transpose(mask_points,0,1)
    sample_points = torch.transpose(mask_points_grid, 1, 2) # [61,441,2], acheived the same effect as origin pymaf_net

    np.save('./data/lspet_mask_points.npy', sample_points)

