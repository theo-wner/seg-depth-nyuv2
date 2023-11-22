#!/bin/bash

##########################################################################################
# Segmentation
##########################################################################################
python test.py --devices 1 --task seg --backbone b0 --checkpoint logs/seg/exp1-backbone/b0/checkpoints/epoch=399-step=40000.ckpt

python test.py --devices 1 --task seg --backbone b1 --checkpoint logs/seg/exp1-backbone/b1/checkpoints/epoch=399-step=40000.ckpt

python test.py --devices 1 --task seg --backbone b2 --checkpoint logs/seg/exp1-backbone/b2/checkpoints/epoch=399-step=40000.ckpt

python test.py --devices 1 --task seg --backbone b3 --checkpoint logs/seg/exp1-backbone/b3/checkpoints/epoch=399-step=40000.ckpt

python test.py --devices 1 --task seg --backbone b4 --checkpoint logs/seg/exp1-backbone/b4/checkpoints/epoch=399-step=40000.ckpt

python test.py --devices 1 --task seg --backbone b5 --checkpoint logs/seg/exp1-backbone/b5/checkpoints/epoch=399-step=40000.ckpt

##########################################################################################
# Depth Estimation
##########################################################################################
python test.py --devices 1 --task depth --backbone b0 --checkpoint logs/depth/exp1-backbone/b0/checkpoints/epoch=399-step=40000.ckpt

python test.py --devices 1 --task depth --backbone b1 --checkpoint logs/depth/exp1-backbone/b1/checkpoints/epoch=399-step=40000.ckpt

python test.py --devices 1 --task depth --backbone b2 --checkpoint logs/depth/exp1-backbone/b2/checkpoints/epoch=399-step=40000.ckpt

python test.py --devices 1 --task depth --backbone b3 --checkpoint logs/depth/exp1-backbone/b3/checkpoints/epoch=399-step=40000.ckpt

python test.py --devices 1 --task depth --backbone b4 --checkpoint logs/depth/exp1-backbone/b4/checkpoints/epoch=399-step=40000.ckpt

python test.py --devices 1 --task depth --backbone b5 --checkpoint logs/depth/exp1-backbone/b5/checkpoints/epoch=399-step=40000.ckpt


##########################################################################################
# Simultaneous Segmentation and Depth Estimation
##########################################################################################
python test.py --devices 1 --task segdepth --backbone b0 --checkpoint logs/segdepth/exp1-backbone/b0/checkpoints/epoch=399-step=40000.ckpt

python test.py --devices 1 --task segdepth --backbone b1 --checkpoint logs/segdepth/exp1-backbone/b1/checkpoints/epoch=399-step=40000.ckpt

python test.py --devices 1 --task segdepth --backbone b2 --checkpoint logs/segdepth/exp1-backbone/b2/checkpoints/epoch=399-step=40000.ckpt

python test.py --devices 1 --task segdepth --backbone b3 --checkpoint logs/segdepth/exp1-backbone/b3/checkpoints/epoch=399-step=40000.ckpt

python test.py --devices 1 --task segdepth --backbone b4 --checkpoint logs/segdepth/exp1-backbone/b4/checkpoints/epoch=399-step=40000.ckpt

python test.py --devices 1 --task segdepth --backbone b5 --checkpoint logs/segdepth/exp1-backbone/b5/checkpoints/epoch=399-step=40000.ckpt


echo "All training runs finished."



