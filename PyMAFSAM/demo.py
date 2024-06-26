# -*- coding: utf-8 -*-
# This script is borrowed and extended from https://github.com/mkocabas/VIBE/blob/master/demo.py and https://github.com/nkolot/SPIN/blob/master/demo.py
# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is
# holder of all proprietary rights on this computer program.
# You can only use this computer program if you have closed
# a license agreement with MPG or you get the right to use the computer
# program from someone who is authorized to grant you that right.
# Any use of the computer program without a valid license is prohibited and
# liable to prosecution.
#
# Copyright©2019 Max-Planck-Gesellschaft zur Förderung
# der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
# for Intelligent Systems. All rights reserved.
#
# Contact: ps-license@tuebingen.mpg.de

import os
os.environ['PYOPENGL_PLATFORM'] = 'egl'

import math
import cv2
import time
import json
import torch
import joblib
import argparse
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
import os.path as osp
from matplotlib.image import imsave
from skimage.transform import resize
from torchvision.transforms import Normalize

from core.cfgs import cfg, parse_args
from models import pymaf_net, pymafsam_net, SMPL
from core import path_config, constants
from datasets.inference import Inference
from utils.renderer import PyRenderer
from utils.imutils import crop

MIN_NUM_FRAMES = 1

def process_image(imgname, keyword, input_res=224):
    """Read image, do preprocessing and possibly crop it according to the bounding box.
    If there are bounding box annotations, use them to crop the image.
    If no bounding box is specified but openpose detections are available, use them to get the bounding box.
    """
    # input params
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    normalize_img = Normalize(mean=constants.IMG_NORM_MEAN, std=constants.IMG_NORM_STD)
    
    img = cv2.imread(f'images/{imgname}.jpg')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Assume that the person is centerered in the image
    height = img.shape[0]
    width = img.shape[1]
    center = np.array([width // 2, height // 2])
    scale = max(height, width) / 200

    img_np = crop(img, center, scale, (input_res, input_res))
     
    img = img_np.astype(np.float32) / 255.
    img = torch.from_numpy(img).permute(2,0,1)
    norm_img = normalize_img(img.clone())[None]
    return img_np, img, norm_img

def run_image_demo(args):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # ========= Define model ========= #
    if args.sampler == 'uniform':
        model = pymaf_net(path_config.SMPL_MEAN_PARAMS, pretrained=True).to(device)
    elif args.sampler == 'masks':
        model = pymafsam_net(path_config.MASKS, path_config.SMPL_MEAN_PARAMS, pretrained=True).to(device)

    # ========= Load pretrained weights ========= #
    if args.checkpoint is not None:
        checkpoint = torch.load(args.checkpoint)
        model.load_state_dict(checkpoint['model'], strict=False)

    # Load SMPL model
    smpl = SMPL(path_config.SMPL_MODEL_DIR,
                batch_size=1,
                create_transl=False).to(device)
    model.eval()

    # Setup renderer for visualization
    renderer = PyRenderer(resolution=(constants.IMG_RES, constants.IMG_RES))

    # Preprocess input image and generate predictions
    img_np, img, norm_img = process_image(imgname=args.image, keyword=args.keyword, input_res=constants.IMG_RES)
    with torch.no_grad():
        if args.regressor == 'pymaf_net':
            preds_dict, feat_dict = model(norm_img.to(device))
            output = preds_dict['smpl_out'][-1]
            pred_camera = output['theta'][:, :3]
            pred_vertices = output['verts']

    # Calculate camera parameters for rendering
    camera_translation = torch.stack([pred_camera[:,1], pred_camera[:,2], 2*constants.FOCAL_LENGTH/(constants.IMG_RES * pred_camera[:,0] +1e-9)],dim=-1)
    camera_translation = camera_translation[0].cpu().numpy()
    pred_vertices = pred_vertices[0].cpu().numpy()

    img_np = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)

    # Render front-view shape
    save_mesh_path = None
    img_shape = renderer(
                    pred_vertices,
                    img=img_np,
                    cam=pred_camera[0].cpu().numpy(),
                    color_type='purple',
                    mesh_filename=save_mesh_path
                )

    # Render side views
    aroundy = cv2.Rodrigues(np.array([0, np.radians(90.), 0]))[0]
    center = pred_vertices.mean(axis=0)
    rot_vertices = np.dot((pred_vertices - center), aroundy) + center

    # Render side-view shape
    img_shape_side = renderer(
                        rot_vertices,
                        img=np.ones_like(img_np),
                        cam=pred_camera[0].cpu().numpy(),
                        color_type='purple',
                        mesh_filename=save_mesh_path
                    )

    # ========= Save rendered image ========= #
    output_path = os.path.join(args.output_folder, args.image)
    os.makedirs(output_path, exist_ok=True)

    # img_name = os.path.basename(args.img_file).split('.')[0]
    img_name = args.image
    save_name = os.path.join(output_path, img_name)

    cv2.imwrite(save_name + '_smpl.png', img_shape)
    cv2.imwrite(save_name + '_smpl_side.png', img_shape_side)

    print(f'Saved the result image to {output_path}.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', default=None,
                        help="image name")
    parser.add_argument('--keyword', default=None,
                        help="keyword")
    parser.add_argument('--image_folder', type=str, default=None,
                        help='input image folder')
    parser.add_argument('--output_folder', type=str, default='output',
                        help='output folder to write results')
    parser.add_argument('--staf_dir', type=str, default='/home/jd/Projects/2D/STAF',
                        help='path to directory STAF pose tracking method.')
    parser.add_argument('--regressor', type=str, default='pymaf_net',
                        help='Name of the SMPL regressor.')
    parser.add_argument('--sampler', type=str, choices=['uniform', 'masks'], default='masks',
                        help='Point-wise sampling method.')
    parser.add_argument('--cfg_file', type=str, default='configs/pymaf_config.yaml',
                        help='config file path.')
    parser.add_argument('--checkpoint', default=None,
                        help='Path to network checkpoint')
    parser.add_argument('--misc', default=None, type=str, nargs="*",
                        help='other parameters')
    parser.add_argument('--model_batch_size', type=int, default=8,
                        help='batch size for SMPL prediction')
    parser.add_argument('--display', action='store_true',
                        help='visualize the results of each step during demo')
    parser.add_argument('--no_render', action='store_true',
                        help='disable final rendering of output video.')
    parser.add_argument('--with_raw', action='store_true',
                        help='attach raw image.')
    parser.add_argument('--empty_bg', action='store_true',
                        help='render meshes on empty background.')
    parser.add_argument('--sideview', action='store_true',
                        help='render meshes from alternate viewpoint.')
    parser.add_argument('--image_based', action='store_true',
                        help='image based reconstruction.')
    parser.add_argument('--use_gt', action='store_true',
                        help='use the ground truth tracking annotations.')
    parser.add_argument('--anno_file', type=str, default='',
                        help='path to tracking annotation file.')
    parser.add_argument('--render_ratio', type=float, default=1.,
                        help='ratio for render resolution')
    parser.add_argument('--recon_result_file', type=str, default='',
                        help='path to reconstruction result file.')
    parser.add_argument('--pre_load_imgs', action='store_true',
                        help='pred-load input images.')
    parser.add_argument('--save_obj', action='store_true',
                        help='save results as .obj files.')

    args = parser.parse_args()
    parse_args(args)
    if args.keyword is None:
        args.keyword = args.image

    if args.image is not None:
        print('Run demo for a single input image.')
        run_image_demo(args)
