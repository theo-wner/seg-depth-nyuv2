#!/bin/bash

##########################################################################################
# Log Errors
##########################################################################################
logfile="exp3_log.txt"

##########################################################################################
# Segmentation
##########################################################################################

python train.py --task seg --name seg/exp3-lr --version e4 --learning_rate 6e-4 1>/dev/null 2>> "$logfile"
echo "-------------------------------------------------------------------------" >> "$logfile" 2>&1

#python train.py --task seg --name seg/exp3-lr --version e6 --learning_rate 6e-6 1>/dev/null 2>> "$logfile"
#echo "-------------------------------------------------------------------------" >> "$logfile" 2>&1


##########################################################################################
# Depth Estimation
##########################################################################################

#python train.py --task depth --name depth/exp3-lr --version e4 --learning_rate 6e-4 1>/dev/null 2>> "$logfile"
#echo "-------------------------------------------------------------------------" >> "$logfile" 2>&1

#python train.py --task depth --name depth/exp3-lr --version e6 --learning_rate 6e-6 1>/dev/null 2>> "$logfile"
#echo "-------------------------------------------------------------------------" >> "$logfile" 2>&1


##########################################################################################
# Simultaneous Segmentation and Depth Estimation
##########################################################################################

python train.py --task segdepth --name segdepth/exp3-lr --version e4 --learning_rate 6e-4 1>/dev/null 2>> "$logfile"
echo "-------------------------------------------------------------------------" >> "$logfile" 2>&1

#python train.py --task segdepth --name segdepth/exp3-lr --version e6 --learning_rate 6e-6 1>/dev/null 2>> "$logfile"



