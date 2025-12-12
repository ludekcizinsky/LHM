#!/bin/bash
set -e # exit on error
# This script runs the full LHM pipeline: data preparation and inference.
# Example usage:
#   bash preprocess.sh taichi 

# activate conda environment
source /home/cizinsky/miniconda3/etc/profile.d/conda.sh
module load gcc ffmpeg

# navigate to project directory
cd /home/cizinsky/LHM

# configurable settings
seq_name=$1

# for now default settings
default_ref_frame_idx=0
frame_path=/scratch/izar/cizinsky/people_snapshot_public/male-3-casual/frames
video_path=$frame_path/video.mp4

# derived paths
preprocess_dir=/scratch/izar/cizinsky/thesis/preprocessing/$seq_name
mkdir -p $preprocess_dir
output_dir=$preprocess_dir/lhm
mkdir -p $output_dir
frame_folder=$output_dir/frames
mkdir -p $frame_folder
initial_gs_model_dir=$output_dir/initial_scene_recon
mkdir -p $initial_gs_model_dir

# echo "--- [1/?] Extracting frames from input video at 30 fps"
# ffmpeg -framerate 30 -start_number 1 \
  # -i $frame_path/frame_%05d.png \
  # -c:v libx264 -pix_fmt yuv420p -movflags +faststart \
  # $video_path

# echo "--- [2/?] Running preprocess.sh to generate motion sequences"
# cd /home/cizinsky/human3r
# bash run_inference.sh $seq_name
# cd /home/cizinsky/LHM

# TODO: it can happen that sam3 actually fails to detect any humans in the scene, so here I would also need to check if everything went fine.
# TODO: it can happen that sam3 will fail to detect certain human for a subset of the frames, so be aware of that
# echo "--- [2/?] Running SAM3 to generate masks and masked images"
# conda deactivate && conda activate sam3
# python engine/new_segment_api/run.py --frames $frame_folder --text "a person" --output-dir $output_dir --prompt-frame 0

echo "[3/?] Running Depth Anything 3 to generate depth maps"
conda deactivate && conda activate da3
python engine/depth_est_api/run.py --output_dir $output_dir

# TODO: manual inspection needed at this point and making sure that mask track ids match motion track ids.
# TODO: another todo is to pick a frame index for each person track to be used as reference frame during inference.
# TODO: I need to ensure I am running over all humans detected in the scene.
# echo "--- [4/?] Running inference.sh to obtain canonical 3dgs models for each human"
# conda deactivate && conda activate lhm
# bash inference.sh $seq_name 0 $default_ref_frame_idx LHM-1B
# bash inference.sh $seq_name 1 $default_ref_frame_idx LHM-1B