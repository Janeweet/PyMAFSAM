"""
This script is borrowed and modified from https://github.com/HongwenZhang/PyMAF/core/path_config.py
path configuration
This file contains definitions of useful data stuctures and the paths
for the datasets and data files necessary to run the code.
Things you need to change: *_ROOT that indicate the path to each dataset
"""
from os.path import join, expanduser

# H36M_ROOT = join('/root/autodl-tmp', 'Datasets/h36m')
LSP_ROOT = join('/root/autodl-tmp', 'Datasets/LSP/lsp_dataset_small')
LSP_ORIGINAL_ROOT = join('/root/autodl-tmp', 'Datasets/LSP/lsp_dataset_original')
LSPET_ROOT = join('/root/autodl-tmp', 'Datasets/LSP/hr-lspet')
COCO_ROOT = join('/root/autodl-tmp', 'Datasets/coco')
PW3D_ROOT = join('/root/autodl-tmp', 'Datasets/3DPW')

# Output folder to save test/train npz files
DATASET_NPZ_PATH = 'data/dataset_extras'

# Output folder to store the openpose detections
# This is requires only in case you want to regenerate 
# the .npz files with the annotations.
OPENPOSE_PATH = 'datasets/openpose'

DATASET_FOLDERS = {
    'lsp-orig': LSP_ORIGINAL_ROOT,
    'lsp': LSP_ROOT,
    'lspet': LSPET_ROOT,
    'coco': COCO_ROOT,
    'dp_coco': COCO_ROOT,
    '3dpw': PW3D_ROOT,
    'coco-full': COCO_ROOT
}

DATASET_FILES = [{
    'lsp': join(DATASET_NPZ_PATH, 'lsp_dataset_test.npz'),
    '3dpw': join(DATASET_NPZ_PATH, '3dpw_test_small.npz'),
    'coco': join(DATASET_NPZ_PATH, 'coco_2014_val.npz'),
    'dp_coco': join(DATASET_NPZ_PATH, 'dp_coco_2014_minival.npz'),
},
    {
        '3dpw': join(DATASET_NPZ_PATH, '3dpw_train.npz'),
        'lsp-orig': join(DATASET_NPZ_PATH, 'lsp_dataset_original_train.npz'),
        'coco': join(DATASET_NPZ_PATH, 'coco_2014_train.npz'),
        'coco-full': join(DATASET_NPZ_PATH, 'coco-full_train_eft.npz'),
        # 'lspet': join(DATASET_NPZ_PATH, 'hr-lspet_train_eft.npz'),
        'lspet': join(DATASET_NPZ_PATH, 'hr-lspet_train_small.npz'),
    }
]

CUBE_PARTS_FILE = 'data/cube_parts.npy'
JOINT_REGRESSOR_TRAIN_EXTRA = 'data/J_regressor_extra.npy'
JOINT_REGRESSOR_H36M = 'data/J_regressor_h36m.npy'
VERTEX_TEXTURE_FILE = 'data/vertex_texture.npy'
STATIC_FITS_DIR = 'data/static_fits'
FINAL_FITS_DIR = 'data/final_fits'
SMPL_MEAN_PARAMS = 'data/smpl_mean_params.npz'
SMPL_MODEL_DIR = 'data/smpl'
MASKS = 'data/lspet_mask_points.npy' # mask lists compressed from lspet training data following the dataset order
