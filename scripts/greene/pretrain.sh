#!/bin/bash

#SBATCH --account=pr_136_tandon_advanced
#SBATCH --job-name="pretrain"
#SBATCH --output="pretrain.%j.out"
#SBATCH -N 4
#SBATCH -c 16
#SBATCH --ntasks-per-node=1
#SBATCH --mem=64g
#SBATCH --gres=gpu:1
#SBATCH -t 48:00:00
#SBATCH --constraint=h100|a100
#SBATCH --export=ALL

config_file=$1
start_time=$(date +%s)
echo "Start time: $start_time"
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
    --config configs/pretrain/${config_file}.yaml
"
$accelerate_cmd $script_path $script_args
cd scripts/greene
conda activate mv

end_time=$(date +%s)
echo "End time: $end_time"
time_taken=$((end_time - start_time))
# show time taken in hours, minutes, and seconds
hours=$((time_taken / 3600))
minutes=$(((time_taken % 3600) / 60))
seconds=$((time_taken % 60))
echo "Time taken: ${hours}:$(printf "%02d" $minutes):$(printf "%02d" $seconds)"
