#!/bin/bash

##########################################################################################
# Segmentation
##########################################################################################

python train.py --task seg --name seg/exp2-augmentation --version fff --augmentations none

python train.py --task seg --name seg/exp2-augmentation --version tff --augmentations rand_scale

python train.py --task seg --name seg/exp2-augmentation --version ftf --augmentations rand_flip

python train.py --task seg --name seg/exp2-augmentation --version fft --augmentations colorjitter

python train.py --task seg --name seg/exp2-augmentation --version ttf --augmentations rand_scale rand_flip

python train.py --task seg --name seg/exp2-augmentation --version tft --augmentations rand_scale colorjitter

python train.py --task seg --name seg/exp2-augmentation --version ftt --augmentations rand_flip colorjitter

python train.py --task seg --name seg/exp2-augmentation --version ttt --augmentations rand_scale rand_flip colorjitter


##########################################################################################
# Depth Estimation
##########################################################################################

python train.py --task depth --name depth/exp2-augmentation --version fff --augmentations none

python train.py --task depth --name depth/exp2-augmentation --version tff --augmentations rand_scale

python train.py --task depth --name depth/exp2-augmentation --version ftf --augmentations rand_flip

python train.py --task depth --name depth/exp2-augmentation --version fft --augmentations colorjitter

python train.py --task depth --name depth/exp2-augmentation --version ttf --augmentations rand_scale rand_flip

python train.py --task depth --name depth/exp2-augmentation --version tft --augmentations rand_scale colorjitter

python train.py --task depth --name depth/exp2-augmentation --version ftt --augmentations rand_flip colorjitter

python train.py --task depth --name depth/exp2-augmentation --version ttt --augmentations rand_scale rand_flip colorjitter



##########################################################################################
# Simultaneous Segmentation and Depth Estimation
##########################################################################################

python train.py --task segdepth --name segdepth/exp2-augmentation --version fff --augmentations none

python train.py --task segdepth --name segdepth/exp2-augmentation --version tff --augmentations rand_scale

python train.py --task segdepth --name segdepth/exp2-augmentation --version ftf --augmentations rand_flip

python train.py --task segdepth --name segdepth/exp2-augmentation --version fft --augmentations colorjitter

python train.py --task segdepth --name segdepth/exp2-augmentation --version ttf --augmentations rand_scale rand_flip

python train.py --task segdepth --name segdepth/exp2-augmentation --version tft --augmentations rand_scale colorjitter

python train.py --task segdepth --name segdepth/exp2-augmentation --version ftt --augmentations rand_flip colorjitter

python train.py --task segdepth --name segdepth/exp2-augmentation --version ttt --augmentations rand_scale rand_flip colorjitter

echo "All training runs finished."



