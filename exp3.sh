#!/bin/bash

##########################################################################################
# Log Errors
##########################################################################################
logfile="exp3_log.txt"

##########################################################################################
# Segmentation
##########################################################################################

#python train.py --task seg --name seg/exp3-lr --version smaller --learning_rate 1e-5 --devices 2 1>/dev/null 2>> "$logfile"
#echo "-------------------------------------------------------------------------" >> "$logfile" 2>&1

python train.py --task seg --name seg/exp3-lr --version larger --learning_rate 1e-4 --devices 2 1>/dev/null 2>> "$logfile"
echo "-------------------------------------------------------------------------" >> "$logfile" 2>&1


##########################################################################################
# Depth Estimation
##########################################################################################

#python train.py --task depth --name depth/exp3-lr --version smaller --learning_rate 1e-5 --devices 2 1>/dev/null 2>> "$logfile"
#echo "-------------------------------------------------------------------------" >> "$logfile" 2>&1

python train.py --task depth --name depth/exp3-lr --version larger --learning_rate 1e-4 --devices 2 1>/dev/null 2>> "$logfile"
echo "-------------------------------------------------------------------------" >> "$logfile" 2>&1


##########################################################################################
# Simultaneous Segmentation and Depth Estimation
##########################################################################################

python train.py --task segdepth --name segdepth/exp3-lr --version smaller --learning_rate 1e-5 --devices 2 1>/dev/null 2>> "$logfile"
echo "-------------------------------------------------------------------------" >> "$logfile" 2>&1

python train.py --task segdepth --name segdepth/exp3-lr --version larger --learning_rate 1e-4 --devices 2 1>/dev/null 2>> "$logfile"
echo "-------------------------------------------------------------------------" >> "$logfile" 2>&1



