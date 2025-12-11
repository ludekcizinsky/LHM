#!/bin/bash
set -e # exit on error

# activate conda environment
source /home/cizinsky/miniconda3/etc/profile.d/conda.sh
module load gcc ffmpeg
conda activate lhm

# navigate to project directory
cd /home/cizinsky/LHM

# seq_name="hi4d_pair19_piggyback"
# num_persons=2
# exp_name="dev_v3"
# source_cam_id=4
# target_cam_ids="[28,76]"
# root_gt_dir_path=/scratch/izar/cizinsky/ait_datasets/full/hi4d/pair19_2/piggyback19
# python LHM/train_multi_humans.py scene_name=$seq_name exp_name=$exp_name num_persons=$num_persons nvs_eval.root_gt_dir_path=$root_gt_dir_path nvs_eval.source_camera_id=$source_cam_id nvs_eval.target_camera_ids=$target_cam_ids eval_pretrain=false epochs=20 eval_every_epoch=30

# seq_name="hi4d_pair16_jump"
# num_persons=2
# exp_name="dev_v3"
# source_cam_id=4
# target_cam_ids="[28,76]"
# root_gt_dir_path=/scratch/izar/cizinsky/ait_datasets/full/hi4d/pair16/pair16/fight16
# python LHM/train_multi_humans.py scene_name=$seq_name exp_name=$exp_name num_persons=$num_persons nvs_eval.root_gt_dir_path=$root_gt_dir_path nvs_eval.source_camera_id=$source_cam_id nvs_eval.target_camera_ids=$target_cam_ids eval_pretrain=false epochs=20 eval_every_epoch=30

# seq_name="hi4d_pair17_dance"
# num_persons=2
# exp_name="tune_offset"
# source_cam_id=28
# # target_cam_ids="[4,16,28,40,52,64,76,88]"
# target_cam_ids="[4,28,52]"
# root_gt_dir_path=/scratch/izar/cizinsky/ait_datasets/full/hi4d/pair17_1/pair17/dance17
# python LHM/train_multi_humans.py scene_name=$seq_name exp_name=$exp_name num_persons=$num_persons nvs_eval.root_gt_dir_path=$root_gt_dir_path nvs_eval.source_camera_id=$source_cam_id nvs_eval.target_camera_ids=$target_cam_ids eval_pretrain=true epochs=50 eval_every_epoch=25 wandb.enable=true


seq_name="hi4d_pair17_dance"
num_persons=2
exp_name="tune_all_params_with_reg"
source_cam_id=28
# target_cam_ids="[4,16,28,40,52,64,76,88]"
# train_params="[]"
target_cam_ids="[4,28,52]"
root_gt_dir_path=/scratch/izar/cizinsky/ait_datasets/full/hi4d/pair17_1/pair17/dance17
python LHM/train_multi_humans.py scene_name=$seq_name exp_name=$exp_name num_persons=$num_persons nvs_eval.root_gt_dir_path=$root_gt_dir_path nvs_eval.source_camera_id=$source_cam_id nvs_eval.target_camera_ids=$target_cam_ids eval_pretrain=true epochs=50 eval_every_epoch=25 wandb.enable=true