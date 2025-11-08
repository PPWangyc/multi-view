#!/bin/bash

#SBATCH --job-name="create_3d_ssl"
#SBATCH --output="create_3d_ssl.%j.out"
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128g
#SBATCH --time=48:00:00
#SBATCH --export=ALL

# Script to run VGGT inference on multi-view images to generate pseudo 3D information for SSL training
# 
# Usage:
#   sbatch create_3d_ssl.sh <dataset> [model_name]
#   or
#   bash create_3d_ssl.sh <dataset> [model_name]
#
# Arguments:
#   dataset: Dataset name (e.g., fly-anipose, mirror-mouse-separate)
#   model_name: (Optional) VGGT model name from HuggingFace or local path
#               If not provided, will try to auto-detect from HuggingFace
#
# Example:
#   sbatch create_3d_ssl.sh fly-anipose
#   sbatch create_3d_ssl.sh fly-anipose facebook/vggt-base

dataset=$1
model_name=${2:-""}  # Optional: VGGT model name from HuggingFace

start_time=$(date +%s)
echo "Start time: $start_time"

source ~/.bashrc

source "$HOME/miniconda3/etc/profile.d/conda.sh"

cd ../..

conda activate mv

# Set PYTHONPATH to include src directory so modules can be imported
export PYTHONPATH="${PWD}/src:${PYTHONPATH}"

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
    echo "Using GPU(s): $available_devices"
else
    unset CUDA_VISIBLE_DEVICES
    echo "WARNING: No GPUs available. Falling back to CPU."
fi
# ============================================================================

# Set default paths
# Adjust these paths based on your data directory structure
data_dir="${PWD}/data/ssl/${dataset}"
output_dir="${PWD}/data/ssl/${dataset}"

# Check if data directory exists
if [ ! -d "$data_dir" ]; then
    echo "ERROR: Data directory does not exist: $data_dir"
    echo "Please ensure the dataset has been created using create_mv_ssl.sh first."
    exit 1
fi

echo "============================================================"
echo "VGGT Pseudo 3D Generation for SSL Training"
echo "============================================================"
echo "Dataset: $dataset"
echo "Data directory: $data_dir"
echo "Output directory: $output_dir"
echo "Model: ${model_name:-Auto-detect from HuggingFace}"
echo "GPU(s): ${CUDA_VISIBLE_DEVICES:-CPU}"
echo "=" * 60

# Build command arguments
script_args="
    --data_dir ${data_dir}
    --output_dir ${output_dir}
    --device cuda
    --seed 42
    --model_name facebook/VGGT-1B
"

# Run VGGT inference
echo "Running VGGT inference..."
python src/create_dataset/create_3d_ssl.py $script_args

# Check exit status
if [ $? -eq 0 ]; then
    echo "VGGT inference completed successfully!"
    echo "Output saved to: $output_dir"
else
    echo "ERROR: VGGT inference failed. Please check the logs above."
fi

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

