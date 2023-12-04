#!/bin/bash

##########################################################################################
# Log Errors
##########################################################################################
logfile="exp4_log.txt"

##########################################################################################
#Train
##########################################################################################
python train.py --task segdepth --name segdepth/exp4-loss --version 0713 --loss_seg_weight 0.7 --loss_depth_weight 1.3 1>/dev/null 2>> "$logfile"
echo "-------------------------------------------------------------------------" >> "$logfile" 2>&1

python train.py --task segdepth --name segdepth/exp4-loss --version 0812 --loss_seg_weight 0.8 --loss_depth_weight 1.2 1>/dev/null 2>> "$logfile"
echo "-------------------------------------------------------------------------" >> "$logfile" 2>&1

python train.py --task segdepth --name segdepth/exp4-loss --version 0911 --loss_seg_weight 0.9 --loss_depth_weight 1.1 1>/dev/null 2>> "$logfile"
echo "-------------------------------------------------------------------------" >> "$logfile" 2>&1

python train.py --task segdepth --name segdepth/exp4-loss --version 1109 --loss_seg_weight 1.1 --loss_depth_weight 0.9 1>/dev/null 2>> "$logfile"
echo "-------------------------------------------------------------------------" >> "$logfile" 2>&1

python train.py --task segdepth --name segdepth/exp4-loss --version 1208 --loss_seg_weight 1.2 --loss_depth_weight 0.8 1>/dev/null 2>> "$logfile"
echo "-------------------------------------------------------------------------" >> "$logfile" 2>&1

python train.py --task segdepth --name segdepth/exp4-loss --version 1307 --loss_seg_weight 1.3 --loss_depth_weight 0.7 1>/dev/null 2>> "$logfile"
echo "-------------------------------------------------------------------------" >> "$logfile" 2>&1




