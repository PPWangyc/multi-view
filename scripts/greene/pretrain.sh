#!/bin/bash

#SBATCH --account=pr_136_tandon_advanced
#SBATCH --job-name="train_encoding"
#SBATCH --output="train_encoding.%j.out"
#SBATCH -N 8
#SBATCH -c 16
#SBATCH --ntasks-per-node=1
#SBATCH --mem=64g
#SBATCH --gres=gpu:1
#SBATCH -t 48:00:00
#SBATCH --constraint=a100|h100|rtx8000
#SBATCH --export=ALL

config_file=$1

. ~/.bashrc

cd ../..

conda activate mv

# Get the script path
script_path=src/pretrain.py

# Get the accelerate command from the setup script
accelerate_cmd=$(./scripts/greene/setup_multi_gpu.sh)
echo "Accelerate command is: $accelerate_cmd"

# Script arguments
script_args="
    --config configs/${config_file}.yaml
"
$accelerate_cmd $script_path $script_args
cd scripts/greene
conda activate mv