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

vid_path=/scratch/izar/cizinsky/ait_datasets/full/hi4d/pair00_1/dance00/images/4/video.mp4
seq_name=hi4d_pair00_dance
bash preprocess.sh $seq_name $vid_path

vid_path=/scratch/izar/cizinsky/ait_datasets/full/hi4d/pair00_1/fight00/images/4/video.mp4
seq_name=hi4d_pair00_fight
bash preprocess.sh $seq_name $vid_path