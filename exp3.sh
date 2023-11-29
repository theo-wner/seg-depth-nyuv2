#!/bin/bash

##########################################################################################
# Segmentation
##########################################################################################

python train.py --task seg --name seg/exp3-lr --version e4 --learning_rate 6e-4

python train.py --task seg --name seg/exp3-lr --version e6 --learning_rate 6e-6


##########################################################################################
# Depth Estimation
##########################################################################################

python train.py --task depth --name depth/exp3-lr --version e4 --learning_rate 6e-4

python train.py --task depth --name depth/exp3-lr --version e6 --learning_rate 6e-6


##########################################################################################
# Simultaneous Segmentation and Depth Estimation
##########################################################################################

python train.py --task segdepth --name segdepth/exp3-lr --version e4 --learning_rate 6e-4

python train.py --task segdepth --name segdepth/exp3-lr --version e6 --learning_rate 6e-6

echo "All training runs finished."



