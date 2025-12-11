#!/bin/bash
#SBATCH --job-name=tune_lhm
#SBATCH --output=/scratch/izar/cizinsky/thesis/slurm/%x.%j.out
#SBATCH --error=/scratch/izar/cizinsky/thesis/slurm/%x.%j.err
#SBATCH --time=02:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8           
#SBATCH --mem=32G                   
#SBATCH --gres=gpu:1 
#SBATCH --account=master 

# Change to the directory where the script is located
cd /home/cizinsky/LHM
source /home/cizinsky/miniconda3/etc/profile.d/conda.sh
conda deactivate && conda activate lhm
module load gcc ffmpeg

# --- V1
# exp_name="default"
# bash run_full.sh $seq_name $path_to_video $exp_name $lr $wd $gclip

# exp_name="lr001"
# bash run_full.sh $seq_name $path_to_video $exp_name 0.001 $wd $gclip

# exp_name="lr005"
# bash run_full.sh $seq_name $path_to_video $exp_name 0.005 $wd $gclip

# exp_name="lr01"
# bash run_full.sh $seq_name $path_to_video $exp_name 0.01 $wd $gclip

# exp_name="wd001"
# bash run_full.sh $seq_name $path_to_video $exp_name $lr 0.001 $gclip

# exp_name="wd005"
# bash run_full.sh $seq_name $path_to_video $exp_name $lr 0.005 $gclip

# exp_name="wd01"
# bash run_full.sh $seq_name $path_to_video $exp_name $lr 0.01 $gclip

# exp_name="gclip05"
# bash run_full.sh $seq_name $path_to_video $exp_name $lr $wd 0.05

# exp_name="gclip2"
# bash run_full.sh $seq_name $path_to_video $exp_name $lr $wd 0.2

# exp_name="gclip4"
# bash run_full.sh $seq_name $path_to_video $exp_name $lr $wd 0.4

# --- V2
# seq_name="hi4d_pair00_fight"
# num_persons=2
# exp_name="dev"
# root_gt_dir_path=/scratch/izar/cizinsky/ait_datasets/full/hi4d/pair00_1/fight00
# python LHM/eval_multi_lhm.py scene_name=$seq_name num_persons=$num_persons nvs_eval.root_gt_dir_path=$root_gt_dir_path

# --- LHM eval on hi4d
# seq_name="hi4d_pair15_fight"
# num_persons=2
# exp_name="dev"
# source_cam_id=4
# root_gt_dir_path=/scratch/izar/cizinsky/ait_datasets/full/hi4d/pair15_1/pair15/fight15
# python LHM/eval_multi_lhm.py scene_name=$seq_name num_persons=$num_persons nvs_eval.root_gt_dir_path=$root_gt_dir_path nvs_eval.source_camera_id=$source_cam_id

# seq_name="hi4d_pair16_jump"
# num_persons=2
# exp_name="dev"
# source_cam_id=4
# root_gt_dir_path=/scratch/izar/cizinsky/ait_datasets/full/hi4d/pair16/pair16/jump16
# python LHM/eval_multi_lhm.py scene_name=$seq_name num_persons=$num_persons nvs_eval.root_gt_dir_path=$root_gt_dir_path nvs_eval.source_camera_id=$source_cam_id

# seq_name="hi4d_pair17_dance"
# num_persons=2
# exp_name="dev"
# source_cam_id=28
# root_gt_dir_path=/scratch/izar/cizinsky/ait_datasets/full/hi4d/pair17_1/pair17/dance17
# python LHM/eval_multi_lhm.py scene_name=$seq_name num_persons=$num_persons nvs_eval.root_gt_dir_path=$root_gt_dir_path nvs_eval.source_camera_id=$source_cam_id

# seq_name="hi4d_pair19_piggyback"
# num_persons=2
# exp_name="dev"
# source_cam_id=4
# root_gt_dir_path=/scratch/izar/cizinsky/ait_datasets/full/hi4d/pair19_2/piggyback19
# python LHM/eval_multi_lhm.py scene_name=$seq_name num_persons=$num_persons nvs_eval.root_gt_dir_path=$root_gt_dir_path nvs_eval.source_camera_id=$source_cam_id

# --- LHM finetune
# seq_name="hi4d_pair19_piggyback"
# num_persons=2
# exp_name="v3"
# source_cam_id=4
# target_cam_ids="[4,28,76]"
# root_gt_dir_path=/scratch/izar/cizinsky/ait_datasets/full/hi4d/pair19_2/piggyback19
# python LHM/train_multi_humans.py scene_name=$seq_name exp_name=$exp_name num_persons=$num_persons nvs_eval.root_gt_dir_path=$root_gt_dir_path nvs_eval.source_camera_id=$source_cam_id nvs_eval.target_camera_ids=$target_cam_ids eval_pretrain=true epochs=50 eval_every_epoch=25

seq_name="hi4d_pair17_dance"
num_persons=2
exp_name="bbox_tune_offset_and_scaling_and_color"
source_cam_id=28
# target_cam_ids="[4,16,28,40,52,64,76,88]"
train_params="[offset_xyz,scaling,shs]"
target_cam_ids="[4,28,52]"
root_gt_dir_path=/scratch/izar/cizinsky/ait_datasets/full/hi4d/pair17_1/pair17/dance17
python LHM/train_multi_humans.py scene_name=$seq_name exp_name=$exp_name num_persons=$num_persons nvs_eval.root_gt_dir_path=$root_gt_dir_path nvs_eval.source_camera_id=$source_cam_id nvs_eval.target_camera_ids=$target_cam_ids eval_pretrain=true epochs=50 eval_every_epoch=25 wandb.enable=true train_params=$train_params