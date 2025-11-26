#!/bin/bash
# given pose sequence, generating animation video .
# usage: bash inference.sh <seq_name> <track_idx> <ref_frame_idx> <model_name>
# example: bash inference.sh taichi 0 3 LHM-1B

# activate conda environment
source /home/cizinsky/miniconda3/etc/profile.d/conda.sh
conda activate lhm

# configurable settings
seq_name=$1
track_idx=$2
ref_frame_idx=$3
model_name=$4

# derived paths
preprocess_dir="/scratch/izar/cizinsky/multiply-output/preprocessing/data/$seq_name/lhm"
track_dir=$(printf "%02d" "$track_idx")
frame_file=$(printf "%04d.png" "$ref_frame_idx")
image_input="$preprocess_dir/masked_images/$track_dir/$frame_file"
motion_seqs_dir="$preprocess_dir/motion/$track_dir/smplx_params"
save_dir=$preprocess_dir/inference_results/$track_dir

# default settings 
MOTION_IMG_DIR=None
VIS_MOTION=true
MOTION_IMG_NEED_MASK=true
RENDER_FPS=30
MOTION_VIDEO_READ_FPS=30
EXPORT_VIDEO=True

# inference
echo "INFERENCE VIDEO"
python -m LHM.launch infer.human_lrm model_name=$model_name \
        image_input=$image_input \
        export_video=$EXPORT_VIDEO \
        motion_seqs_dir=$motion_seqs_dir motion_img_dir=$MOTION_IMG_DIR  \
        vis_motion=$VIS_MOTION motion_img_need_mask=$MOTION_IMG_NEED_MASK \
        render_fps=$RENDER_FPS motion_video_read_fps=$MOTION_VIDEO_READ_FPS  save_dir=$save_dir