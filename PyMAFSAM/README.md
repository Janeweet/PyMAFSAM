## Deep Learning based 3D Human Body Reconstruction

## Requirements

- Python 3.8
```
conda create --no-default-packages -n pymafx python=3.8
conda activate pymafx
```

### packages

- [PyTorch](https://www.pytorch.org) tested on version 1.9.0
```
conda install pytorch==1.9.0 torchvision==0.10.0 cudatoolkit=11.1 -c pytorch -c conda-forge
```

- [pytorch3d](https://github.com/facebookresearch/pytorch3d/blob/main/INSTALL.md)
```
pip install "git+https://github.com/facebookresearch/pytorch3d.git@stable"
```
- other packages listed in `requirements.txt`
```
pip install -r requirements.txt
```

### necessary files

> mesh_downsampling.npz & DensePose UV data

- Run the following script to fetch mesh_downsampling.npz & DensePose UV data from other repositories.

```
bash fetch_data.sh
```
> SMPL model files

- Collect SMPL model files from [https://smpl.is.tue.mpg.de](https://smpl.is.tue.mpg.de) and [UP](https://github.com/classner/up/blob/master/models/3D/basicModel_neutral_lbs_10_207_0_v1.0.0.pkl). Rename model files and put them into the `./data/smpl` directory.

> Segment Anything(SAM) model files

- Get SAM model from [https://github.com/facebookresearch/segment-anything.git](https://github.com/facebookresearch/segment-anything.git). Install model and put segment_anything folder under root directory:
```
pip install git+https://github.com/facebookresearch/segment-anything.git
          
```
> Fetch preprocessed data from [SPIN](https://github.com/nkolot/SPIN#fetch-data).

> Fetch final_fits data from [SPIN](https://github.com/nkolot/SPIN#final-fits). [important note: using [EFT](https://github.com/facebookresearch/eft) fits for training is much better. Compatible npz files are available [here](https://cloud.tsinghua.edu.cn/d/635c717375664cd6b3f5)]

> Download the [pre-trained model](https://drive.google.com/drive/folders/1xC-aaJgzEf1lNXPguDa9813XhtIamBSq?usp=sharing) and put it into the `./data/pretrained_model` directory.

> Download the [resnet50-19c8e357.pth](https://github.com/fregu856/deeplabv3/blob/master/pretrained_models/resnet/resnet50-19c8e357.pth) and put it into the `./data/pretrained_model` directory.

After collecting the above necessary files, the directory structure of `./data` is expected as follows.  
```
./data
├── dataset_extras
│   └── .npz files
├── J_regressor_extra.npy
├── J_regressor_h36m.npy
├── mesh_downsampling.npz
├── sam_vit_h_4b8939.pth
├── lspet_mask_points.npy
├── pretrained_model
│   └── PyMAFSAM_model_checkpoint.pt
│   └── resnet50-19c8e357.pth
├── smpl
│   ├── SMPL_FEMALE.pkl
│   ├── SMPL_MALE.pkl
│   └── SMPL_NEUTRAL.pkl
├── smpl_mean_params.npz
├── final_fits
│   └── .npy files
└── UV_data
    ├── UV_Processed.mat
    └── UV_symmetry_transforms.mat
```

## Demo
Run the demo code.
#### For image input:

```
python3 demo.py --checkpoint=data/pretrained_model/PyMAFSAM_model_checkpoint.pt --image COCO_val2014
```

## Evaluation

### COCO Keypoint Localization

1. Download the preprocessed data [coco_2014_val.npz](https://drive.google.com/drive/folders/1R4_Vi4TpCQ26-6_b2PhjTBg-nBxZKjz6?usp=sharing). Put it into the `./data/dataset_extras` directory. 

2. Run the COCO evaluation code.
```
python3 eval_coco.py --checkpoint=data/pretrained_model/PyMAF_model_checkpoint.pt
```

### 3DPW 3D Pose and Shape Estimation

Run the evaluation code. Using `--dataset` to specify the evaluation dataset.
```
# Example usage:
# 3DPW
python3 eval.py --checkpoint=data/pretrained_model/PyMAF_model_checkpoint.pt --dataset=3dpw --log_freq=20
```

## Mask

I use LSP-Extended dataset and build the first 960 image masks as sampling points. The masks are visualized in ./images/lspet_small_mask

### Generation & Store
The mask_save file is for mask prompt customization, mask display, and mask point storage. Foreground and background mask points are saved seperatedly in ./data/masks/
```
python3 mask_saver.py --index 1005
```
### Mask Points
Mask points are mainly selected from the foreground file while the number of points is 441, the same with the baseline. Points are stored at ./data/lspet_mask_points.npy
```
python3 mask_points_selector.y --imgrange 1005
``` 

## Training

Below messages are the training details of the conference version of this work.

To perform training, we need to collect preprocessed files of training datasets at first.

The preprocessed labels have the same format as SPIN and can be retrieved from [here](https://github.com/nkolot/SPIN#fetch-data). Please refer to [SPIN](https://github.com/nkolot/SPIN) for more details about data preprocessing.

My work is trained on single LSP-Extended dataset. Example usage:
```
# training on LSP-Extended
CUDA_VISIBLE_DEVICES=0 nohup python3 train.py --regressor pymaf_net --single_dataset --single_dataname lspet --misc TRAIN.BATCH_SIZE 32
```
We can monitor the training process by setting up a TensorBoard at the directory `./logs`. Go into the parent foler of log summary is you want to visualize the logs.
```
# display the log
tensorboard --logdir=path/to/summary/folder
```

## Acknowledgments

The code is developed upon the following projects. Many thanks to their contributions.
- [PyMAF](https://github.com/HongwenZhang/PyMAF)

- [SAM](https://github.com/facebookresearch/segment-anything)

- [HMR](https://github.com/akanazawa/hmr)

- [pose_resnet](https://github.com/Microsoft/human-pose-estimation.pytorch)
