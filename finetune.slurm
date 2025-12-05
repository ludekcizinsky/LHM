#!/bin/bash
#SBATCH --job-name=lhm
#SBATCH --output=/scratch/izar/cizinsky/thesis/slurm/%x.%j.out
#SBATCH --error=/scratch/izar/cizinsky/thesis/slurm/%x.%j.err
#SBATCH --time=01:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8           
#SBATCH --mem=32G                   
#SBATCH --gres=gpu:1 
#SBATCH --account=master 

# Change to the directory where the script is located
cd /home/cizinsky/LHM
conda deactivate && conda activate lhm

# --- Shared settings
seq_name="hi4d_pair15_fight"
num_persons=2

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
exp_name="dev"
python LHM/finetune_multi_humans.py scene_name=$seq_name num_persons=$num_persons