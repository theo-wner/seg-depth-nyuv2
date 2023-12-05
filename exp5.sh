#!/bin/bash

##########################################################################################
# Log Errors
##########################################################################################
logfile="exp5_ignore.txt"

##########################################################################################
# Train
##########################################################################################
python train.py --task segdepth --name segdepth/exp5-ignore --version ignore_depth --loss_seg_weight 1.0 --loss_depth_weight 0.0 --devices 2 1>/dev/null 2>> "$logfile"
echo "-------------------------------------------------------------------------" >> "$logfile" 2>&1

python train.py --task segdepth --name segdepth/exp5-ignore --version ignore_seg --loss_seg_weight 0.0 --loss_depth_weight 1.0 --devices 2 1>/dev/null 2>> "$logfile"
echo "-------------------------------------------------------------------------" >> "$logfile" 2>&1




