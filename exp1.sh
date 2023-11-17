#!/bin/bash

<< comment

##########################################################################################
# Segmentation
##########################################################################################

echo "b0..."
python train.py --task seg --name seg/exp1-backbone --version b0 --devices 1 --backbone b0

echo "b1..."
python train.py --task seg --name seg/exp1-backbone --version b1 --devices 1 --backbone b1

echo "b2..."
python train.py --task seg --name seg/exp1-backbone --version b2 --devices 1 --backbone b2

echo "b3..."
python train.py --task seg --name seg/exp1-backbone --version b3 --devices 1 --backbone b3

echo "b4..."
python train.py --task seg --name seg/exp1-backbone --version b4 --devices 1 --backbone b4

echo "b5..."
python train.py --task seg --name seg/exp1-backbone --version b5 --devices 1 --backbone b5

comment


##########################################################################################
# Depth Estimation
##########################################################################################

echo "b0..."
python train.py --task depth --name depth/exp1-backbone --version b0 --devices 1 --backbone b0

echo "b1..."
python train.py --task depth --name depth/exp1-backbone --version b1 --devices 1 --backbone b1

echo "b2..."
python train.py --task depth --name depth/exp1-backbone --version b2 --devices 1 --backbone b2

echo "b3..."
python train.py --task depth --name depth/exp1-backbone --version b3 --devices 1 --backbone b3

echo "b4..."
python train.py --task depth --name depth/exp1-backbone --version b4 --devices 1 --backbone b4

echo "b5..."
python train.py --task depth --name depth/exp1-backbone --version b5 --devices 1 --backbone b5


##########################################################################################
# Simultaneous Segmentation and Depth Estimation
##########################################################################################

echo "b0..."
python train.py --task segdepth --name segdepth/exp1-backbone --version b0 --devices 1 --backbone b0

echo "b1..."
python train.py --task segdepth --name segdepth/exp1-backbone --version b1 --devices 1 --backbone b1

echo "b2..."
python train.py --task segdepth --name segdepth/exp1-backbone --version b2 --devices 1 --backbone b2

echo "b3..."
python train.py --task segdepth --name segdepth/exp1-backbone --version b3 --devices 1 --backbone b3

echo "b4..."
python train.py --task segdepth --name segdepth/exp1-backbone --version b4 --devices 1 --backbone b4

echo "b5..."
python train.py --task segdepth --name segdepth/exp1-backbone --version b5 --devices 1 --backbone b5


echo "All training runs finished."



