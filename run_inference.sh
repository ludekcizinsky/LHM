#!/bin/bash

source /home/cizinsky/miniconda3/etc/profile.d/conda.sh
conda activate lhm
module load gcc ffmpeg

scene_dir=$1
ref_frame_idx=$2
model_name=LHM-1B

repo_path=/home/cizinsky/master-thesis
cd $repo_path/submodules/lhm

python inference.py model_name=$model_name scene_dir=$scene_dir input_image_idx=$ref_frame_idx 