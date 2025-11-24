#!/bin/bash

#SBATCH --job-name="dlc"
#SBATCH --output="dlc.%j.out"
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128g
#SBATCH --time=48:00:00
#SBATCH --export=ALL
#SBATCH --priority=999999

# dataset: mirror-mouse, mirror-fish, ibl-paw, crim13
dataset=$1
# train_frames: 75, 100, 800 (dataset dependent)
train_frames=$2
# gpu_id: 0, 1, 2, etc. (optional, defaults to 0)
gpu_id=${3:-0}

start_time=$(date +%s)
echo "Start time: $start_time"

source "$HOME/miniconda3/etc/profile.d/conda.sh"

cd ..
pwd=$(pwd)

conda activate mv
export PYTHONPATH="${pwd}/src:${PYTHONPATH}"

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
    echo "Available GPUs: $available_devices"
    export CUDA_VISIBLE_DEVICES="$available_devices"
    # Use the first available GPU for gpu_id if not explicitly set
    if [ "$gpu_id" = "0" ] && [ -n "$available_devices" ]; then
        gpu_id=$(echo "$available_devices" | cut -d',' -f1)
    fi
else
    unset CUDA_VISIBLE_DEVICES
    echo "WARNING: No GPUs available. Falling back to CPU."
    gpu_id=0
fi

echo "Dataset: $dataset"
echo "Train frames: $train_frames"
echo "GPU ID: $gpu_id"

# Run DLC training script
python dlc/run_dlc.py \
    --dataset "$dataset" \
    --gpu_id "$gpu_id" \
    --train_frames "$train_frames"

cd dlc
conda deactivate

end_time=$(date +%s)
echo "End time: $end_time"
time_taken=$((end_time - start_time))
# show time taken in hours, minutes, and seconds
hours=$((time_taken / 3600))
minutes=$(((time_taken % 3600) / 60))
seconds=$((time_taken % 60))
echo "Time taken: ${hours}:$(printf "%02d" $minutes):$(printf "%02d" $seconds)"

cd dlc