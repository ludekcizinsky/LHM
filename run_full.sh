#!/bin/bash
set -e # exit on error
# This script runs the full LHM pipeline: data preparation and inference.
# Example usage:
#   bash run_full.sh taichi /home/cizinsky/in_the_wild/bilibili/taichi.mp4 0

# activate conda environment
source /home/cizinsky/miniconda3/etc/profile.d/conda.sh
module load gcc ffmpeg

# navigate to project directory
cd /home/cizinsky/LHM

# configurable settings
seq_name=$1
input_video=$2
exp_name=$3


# for now default settings
default_ref_frame_idx=0

# derived paths
preprocess_dir=/scratch/izar/cizinsky/thesis/preprocessing/$seq_name
mkdir -p $preprocess_dir
output_dir=$preprocess_dir/lhm
mkdir -p $output_dir
frame_folder=$output_dir/frames
mkdir -p $frame_folder
initial_gs_model_dir=$output_dir/initial_scene_recon
mkdir -p $initial_gs_model_dir
refined_gs_model_dir=$output_dir/refined_scene_recon
mkdir -p $refined_gs_model_dir

# echo "--- [1/?] Running preprocess.sh to generate motion sequences"
# conda deactivate && conda activate lhm
# bash preprocess.sh $input_video $output_dir

# # TODO: it can happen that sam3 actually fails to detect any humans in the scene, so here I would also need to check if everything went fine.
# echo "--- [2/?] Running SAM3 to generate masks and masked images"
# conda deactivate && conda activate sam3
# python engine/new_segment_api/run.py --frames $frame_folder --text "person" --output-dir $output_dir

# TODO: manual inspection needed at this point and making sure that mask track ids match motion track ids.
# TODO: another todo is to pick a frame index for each person track to be used as reference frame during inference.
# TODO: I need to ensure I am running over all humans detected in the scene.
# echo "--- [3/?] Running inference.sh to obtain canonical 3dgs models for each human"
# conda deactivate && conda activate lhm
# bash inference.sh $seq_name 0 $default_ref_frame_idx LHM-1B
# bash inference.sh $seq_name 1 $default_ref_frame_idx LHM-1B
# TODO: add script that will visualize each canonical human gs in 3D

# echo "--- [4/?] Running inference for multi-human LHM"
# conda deactivate && conda activate lhm
# python LHM/infer_multi_humans.py --gs_model_dir=$initial_gs_model_dir --scene_name=$seq_name --nv_rot_degree=0
# python LHM/infer_multi_humans.py --gs_model_dir=$initial_gs_model_dir --scene_name=$seq_name --nv_rot_degree=90

echo "--- [5/?] Running finetuning for multi-human LHM"
conda deactivate && conda activate lhm
python LHM/finetune_multi_humans.py output_dir=$output_dir scene_name=$seq_name exp_name=$exp_name 

# echo "--- [6/?] Running inference for multi-human LHM"
# conda deactivate && conda activate lhm
# python LHM/infer_multi_humans.py --gs_model_dir=$refined_gs_model_dir/$exp_name --scene_name=$seq_name --nv_rot_degree=0
# python LHM/infer_multi_humans.py --gs_model_dir=$refined_gs_model_dir/$exp_name --scene_name=$seq_name --nv_rot_degree=90