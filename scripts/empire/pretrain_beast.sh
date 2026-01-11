#!/bin/bash

#SBATCH --job-name="pretrain_beast"
#SBATCH --output="pretrain_beast.%j.out"
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=8
#SBATCH --gpus-per-task=8
#SBATCH --cpus-per-task=128
#SBATCH --mem=256g
#SBATCH --time=48:00:00
#SBATCH --account columbia
#SBATCH --partition columbia
#SBATCH --export=ALL



source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate beast

config_file=/mnt/lustre/columbia/ywang1/Projects/multi-view/configs/pretrain/beast_aleks.yaml

beast train -c $config_file