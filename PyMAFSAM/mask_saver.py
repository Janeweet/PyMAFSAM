from utils.seed import seed_everything
from segment_anything import sam_model_registry, SamPredictor
import sys
import torch
import cv2
import os
import argparse
import numpy as np
from PIL import Image
from pathlib import Path
from core.cfgs import cfg, parse_args
from core import path_config, constants
from torchvision.transforms import Normalize
import matplotlib.pyplot as plt
from image_preparer import rgb_preprocess

class Mask():

    def __init__(self, index):
        self.img, self.imgname = rgb_preprocess(index)
        self.imgname = self.imgname[0:7]
        # self.input_point = point

    def show_mask(self, mask, ax, random_color=False):
        if random_color:
            color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
        else:
            color = np.array([30/255, 144/255, 255/255, 0.6])
        h, w = mask.shape[-2:]
        mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
        ax.imshow(mask_image)

    def show_points(self, coords, labels, ax, marker_size=375):
        pos_points = coords[labels==1]
        neg_points = coords[labels==0]
        ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
        ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)

    def show_box(self, box, ax):
        x0, y0 = box[0], box[1]
        w, h = box[2] - box[0], box[3] - box[1]
        ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='red', facecolor=(0,0,0,0), lw=2))

    def mask_predictor(self):
        # input params
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        normalize_img = Normalize(mean=constants.IMG_NORM_MEAN, std=constants.IMG_NORM_STD)

        img = self.img
        imgname = self.imgname

        sam_checkpoint = "./data/sam_vit_h_4b8939.pth"
        sam = sam_model_registry['vit_h'](checkpoint=sam_checkpoint)
        sam.to(device=device)
        
        predictor = SamPredictor(sam)
        predictor.set_image(img)

        # Change prompts and lables for each instances
        input_box = np.array([0,20,224,180])
        input_point = np.array([[50,150],[100,125],[160,100]])
        input_label = np.array([1,1,1])

        masks, _, _ = predictor.predict(
            point_coords=input_point,
            point_labels=input_label,
            box=input_box,
            multimask_output=False
        )
        mask = masks[0]

        mask_fore_point_list = {}
        mask_fore_point_list['imgname'] = []
        mask_fore_point_list['mask_fore_points_x'] = []
        mask_fore_point_list['mask_fore_points_y'] = []

        mask_back_point_list = {}
        mask_back_point_list['imgname'] = []
        mask_back_point_list['mask_back_points_x'] = []
        mask_back_point_list['mask_back_points_y'] = []
        
        mask_fore_point_list['imgname'].append(imgname)
        for i in range(len(mask)):
            for j in range(len(mask[i])):
                if(mask[i][j] == True):
                    mask_fore_point_list['mask_fore_points_x'].append(j)
                    mask_fore_point_list['mask_fore_points_y'].append(i)
                if(mask[i][j] == False):
                    mask_back_point_list['mask_back_points_x'].append(j)
                    mask_back_point_list['mask_back_points_y'].append(i)

        return mask, mask_fore_point_list, mask_back_point_list

    def forePointsDisplay(self, mask_point_list):
        fore_x = mask_point_list['mask_fore_points_x']
        fore_y = mask_point_list['mask_fore_points_y']

        # Visualize mask points
        plt.switch_backend('agg')
        plt.figure('Image')
        plt.imshow(self.img)
        plt.scatter(fore_x, fore_y, c='g', s=1)
        plt.title(self.imgname)
    
        mask_fore_file_path = (Path(f'./images/{self.imgname}/{self.imgname}_mask_fore_points.png'))
        plt.savefig(mask_fore_file_path)


    def maskDisplay(self, mask):

        # Show masks and prompt
        plt.switch_backend('agg')
        plt.figure('Image')
        plt.imshow(self.img)
        plt.title(self.imgname)
        self.show_mask(mask, plt.gca())
        # self.show_points(self.input_point, input_label, plt.gca())
        # mask_obj.show_box(args.input_box, plt.gca())
        plt.axis('off')
        plt.show()
        
        path = f'./images/{self.imgname}'
        if not os.path.exists(path):
            os.makedirs(path)

        mask_file_path = (Path(f'/root/autodl-tmp/PyMAF/images/{self.imgname}/{self.imgname}_masked.png'))
        plt.savefig(mask_file_path)
    
    def pointSaver(self,mask_fore_point_list, mask_back_point_list):
        # Save foreground mask points
        fore_file_path = (Path(f'/root/autodl-tmp/PyMAF/data/masks/masks_fore1_{self.imgname}.npy'))
        ff = open(fore_file_path,'w')
        for k, v in mask_fore_point_list.items():
            ff.write(k+':'+str(v)+'\n')
        ff.close()
        np.save(fore_file_path, mask_fore_point_list)

        # background
        back_file_path = (Path(f'/root/autodl-tmp/PyMAF/data/masks/masks_back1_{self.imgname}.npy'))
        bf = open(back_file_path,'w')
        for k, v in mask_back_point_list.items():
            bf.write(k+':'+str(v)+'\n')
        bf.close()
        np.save(back_file_path, mask_back_point_list)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument('--imgname', default=None,
    #                 help="image name")
    parser.add_argument('--index', type=int,
                    help="image list index in small .npz file")
    parser.add_argument('--cfg_file', type=str, default='configs/pymaf_config.yaml',
                        help='config file path.')
    parser.add_argument('--misc', default=None, type=str, nargs="*",
                    help='other parameters')
    parser.add_argument('--checkpoint', default=None,
                        help='Path to network checkpoint')
    parser.add_argument('--seed', default=0,
                        help="random seed")
    parser.add_argument("--input_box", type=float, nargs='+',
                        help="The area of the box prompt, [upper_L lower_R].")
    parser.add_argument("--point_f_coords", type=float, nargs='+',
                        help="The coordinate of the foreground point prompt, [coord_W coord_H].")
    parser.add_argument("--point_f_add_coords", type=float, nargs='+',
                        help="The coordinate of an additional foreground point prompt, [coord_W coord_H].")
    parser.add_argument("--point_b_coords", type=float, nargs='+',
                        help="The coordinate of the background point prompt, [coord_W coord_H].")
    parser.add_argument("--point_b_add_coords", type=float, nargs='+',
                        help="The coordinate of the background point prompt, [coord_W coord_H].")
    parser.add_argument("--point_coords", type=float, nargs='+',
                        help="The coordinate of the background point prompt, [coord_W coord_H].")
    # parser.add_argument('--plot', action='store_true',
    #                 help='prompt point display flag')
    # parser.add_argument('--imgprep', action='store_true',
    #                 help='image preparation')
    # parser.add_argument('--masksave', action='store_true',
    #                 help='display and save mask')

    args = parser.parse_args()
    parse_args(args)

    mask_obj = Mask(args.index)

    # parser.add_argument('--keyword', default=None,
    #                 help="keyword")

    # if args.imgprep:
    #     imagePreparer(imgname=args.imgname,point_f = args.point_f_coords, point_f_add = args.point_f_add_coords, point_b = args.point_b_coords,)
    
    # if args.masksave:
    # mask, mask_fore_point_list, mask_back_point_list= mask_obj.mask_predictor(point_f = args.point_f_coords, point_f_add = args.point_f_add_coords, point_b = args.point_b_coords, point_b_add = args.point_b_add_coords, random_seed=args.seed)
    # print(masks[0].shape)
    # mask, mask_fore_point_list, mask_back_point_list= mask_obj.mask_predictor(point_f = args.point_f_coords, point_f_add = args.point_f_add_coords, point_b = args.point_b_coords, random_seed=args.seed)
    # 
    mask, mask_fore_point_list, mask_back_point_list= mask_obj.mask_predictor()
    mask_obj.maskDisplay(mask)
    mask_obj.forePointsDisplay(mask_fore_point_list)
    # mask_obj.pointSaver(mask_fore_point_list,mask_back_point_list)

