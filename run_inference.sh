#!/bin/bash
set -e # exit on error
# given pose sequence, generating animation video .
# usage: bash inference.sh <seq_name> <ref_frame_idx>
# example: bash inference.sh taichi 0 

# activate conda environment
source /home/cizinsky/miniconda3/etc/profile.d/conda.sh
conda activate lhm
module load gcc ffmpeg

# configurable settings
seq_name=$1
ref_frame_idx=$2
hi4d_gt_root_dir=$3

# derived paths
preprocess_dir=/scratch/izar/cizinsky/thesis/preprocessing/$seq_name

# defaults
model_name=LHM-1B

# inference
cd submodules/lhm
python inference.py model_name=$model_name output_dir=$preprocess_dir input_image_id=$ref_frame_idx hi4d_gt_root_dir=$hi4d_gt_root_dir