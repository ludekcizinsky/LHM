#!/bin/bash
set -e # exit on error
# This script runs the full LHM pipeline: data preparation and inference.
# usage: bash run_full.sh <seq_name> <nv_rot_degree>
# example: bash run.sh taichi 0

# activate conda environment
source /home/cizinsky/miniconda3/etc/profile.d/conda.sh
module load gcc ffmpeg

# configurable settings
seq_name=$1
nv_rot_degree=$2

# derived paths
preprocess_dir=/scratch/izar/cizinsky/multiply-output/preprocessing/data/$seq_name
output_dir=$preprocess_dir/lhm
frame_folder=$preprocess_dir/image

# navigate to project directory
cd /home/cizinsky/LHM

# ------------ Start of the pipeline
# # TODO: I currently assume that I have already ran sam2, need to add that step here.
# echo "--- [1/4] Running SAM2 to generate masks and masked images"
# input_mask_dir=$preprocess_dir/sam2_masks
# python get_masks_using_sam2.py \
    # --input_mask_dir $input_mask_dir \
    # --input_img_dir $frame_folder \
    # --output_dir $output_dir \
    # --threshold 0.5

echo "--- [1/4] Running SAM3 to generate masks and masked images"
conda deactivate && conda activate sam3
python engine/new_segment_api/run.py

# echo "--- [2/4] Generating video at 10 fps from frames"
# output_video=$preprocess_dir/seq_vid_at10fps.mp4
# ffmpeg -framerate 10 -start_number 0 -i "$frame_folder/%04d.png" \
  # -c:v libx264 -pix_fmt yuv420p -crf 18 $output_video

# echo "--- [3/4] Running preprocess.sh to generate motion sequences"
# conda deactivate && conda activate lhm
# bash preprocess.sh $output_video $output_dir

# TODO: manual inspection needed at this point and making sure that mask track ids match motion track ids.
# TODO: another todo is to pick a frame index for each person track to be used as reference frame during inference.
# echo "--- [4/4] Running inference.sh to generate animations"
# cd /home/cizinsky/LHM
# conda deactivate && conda activate lhm
# default_ref_frame_idx=0
# bash inference.sh $seq_name 0 $default_ref_frame_idx LHM-1B
# bash inference.sh $seq_name 1 $default_ref_frame_idx LHM-1B
# gs_model_dir=/scratch/izar/cizinsky/multiply-output/preprocessing/data/$seq_name/lhm/inference_results
# save_dir=/scratch/izar/cizinsky/thesis/evaluation/videos/renders/custom/lhm
# python LHM/infer_multi_humans.py --gs_model_dir=$gs_model_dir --save_dir=$save_dir --scene_name=$seq_name --nv_rot_degree=$nv_rot_degree

# echo "--- [5/4] Running finetuning for multi-human LHM"
# cd /home/cizinsky/LHM
# conda deactivate && conda activate lhm
# gs_model_dir=/scratch/izar/cizinsky/multiply-output/preprocessing/data/$seq_name/lhm/inference_results
# save_dir=/scratch/izar/cizinsky/thesis/evaluation/videos/renders/custom/lhm_finetuned
# python LHM/finetune_multi_humans.py --gs_model_dir=$gs_model_dir --save_dir=$save_dir --scene_name=$seq_name --nv_rot_degree=$nv_rot_degree