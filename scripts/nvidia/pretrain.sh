#!/bin/bash

#SBATCH --job-name="pretrain"
#SBATCH --output="pretrain.%j.out"
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=8
#SBATCH --gpus-per-task=8
#SBATCH --cpus-per-task=32
#SBATCH --mem=256g
#SBATCH --time=UNLIMITED
#SBATCH --export=ALL

config_file=$1
start_time=$(date +%s)
echo "Start time: $start_time"
source ~/.bashrc

source "$HOME/miniconda3/etc/profile.d/conda.sh"

cd ../..

conda activate mv

# ============================================================================
# GPU Setup (Reusable pattern - copy this block to other scripts)
# ============================================================================
# Check available GPUs and configure CUDA_VISIBLE_DEVICES
# This script handles both SLURM and non-SLURM environments
gpu_info=$(scripts/nvidia/check_avail_gpu.sh)
num_processes=$(echo "$gpu_info" | head -n 1)
available_devices=$(echo "$gpu_info" | tail -n 1)

# Set CUDA_VISIBLE_DEVICES to available GPUs (or unset if no GPUs available)
if [ -n "$available_devices" ] && [ "$num_processes" -ge 1 ]; then
    export CUDA_VISIBLE_DEVICES="$available_devices"
else
    unset CUDA_VISIBLE_DEVICES
    echo "WARNING: No GPUs available. Falling back to CPU."
fi

# Get accelerate command (respects CUDA_VISIBLE_DEVICES if set)
if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
    accelerate_cmd=$(CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" ./scripts/setup_multi_gpu.sh)
else
    accelerate_cmd=$(./scripts/setup_multi_gpu.sh)
fi
# ============================================================================

# Script configuration
script_path=src/pretrain.py
script_args="
    --config configs/pretrain/${config_file}.yaml
"
echo "Accelerate command: $accelerate_cmd"
$accelerate_cmd $script_path $script_args

cd scripts/nvidia
conda deactivate

end_time=$(date +%s)
echo "End time: $end_time"
time_taken=$((end_time - start_time))
# show time taken in hours, minutes, and seconds
hours=$((time_taken / 3600))
minutes=$(((time_taken % 3600) / 60))
seconds=$((time_taken % 60))
echo "Time taken: ${hours}:$(printf "%02d" $minutes):$(printf "%02d" $seconds)"

