#!/bin/bash
# usage: bash preprocess.sh /scratch/izar/cizinsky/multiply-output/preprocessing/data/taichi/taichi_10fps.mp4
source /home/cizinsky/miniconda3/etc/profile.d/conda.sh
conda activate lhm

cd /home/cizinsky/LHM

video_path=$1
output_path=/home/cizinsky/LHM/train_data/custom_motion
mkdir -p $output_path
python engine/pose_estimation/video2motion.py --video_path $video_path --output_path $output_path --visualize
