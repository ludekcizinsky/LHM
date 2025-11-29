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

# --- Shared settings
seq_name="taichi"
path_to_video="/home/cizinsky/in_the_wild/bilibili/taichi.mp4"

# --- Expriments
exp_name="dev"
bash run_full.sh $seq_name $path_to_video $exp_name