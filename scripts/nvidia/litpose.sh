#!/bin/bash

#SBATCH --job-name="litpose"
#SBATCH --output="litpose.%j.out"
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=256g
#SBATCH --time=48:00:00
#SBATCH --export=ALL

# fly-anipose, mirror-mouse-separate
dataset=$1
# resnet50_animal_ap10k, vitb_imagenet, vits_dino, vitb_dino | Use pretrained model
# vitb-mv, vitb-sv, vitb-mvt, vits-mvt | Need to load pretrained model weights from checkpoint
model=$2
# prediction model type: sv, mv
model_type=$3
# train frame: 100, 1000
train_frame=$4 
# seed: 0, 1, 2
seed=$5 

if [ "$model_type" = "sv" ]; then
    model_type="heatmap"
elif [ "$model_type" = "mv" ]; then
    model_type="heatmap_multiview_transformer"
fi

start_time=$(date +%s)
echo "Start time: $start_time"

source "$HOME/miniconda3/etc/profile.d/conda.sh"

cd ../..

conda activate mv
pwd=$(pwd)
export PYTHONPATH="${pwd}/src:${PYTHONPATH}"

# Set default values
epochs=300
mode=ft
data_dir=${pwd}/data/${dataset}

# ============================================================================
# GPU Setup (Reusable pattern - copy this block to other scripts)
# ============================================================================
# Check available GPUs and configure CUDA_VISIBLE_DEVICES
gpu_info=$(scripts/nvidia/check_avail_gpu.sh)
num_processes=$(echo "$gpu_info" | head -n 1)
available_devices=$(echo "$gpu_info" | tail -n 1)

# Set CUDA_VISIBLE_DEVICES to available GPUs (or unset if no GPUs available)
if [ -n "$available_devices" ] && [ "$num_processes" -ge 1 ]; then
    export CUDA_VISIBLE_DEVICES="$available_devices"
    echo "Using $num_processes GPU(s): $available_devices"
else
    unset CUDA_VISIBLE_DEVICES
    echo "WARNING: No GPUs available. Falling back to CPU."
fi

echo "Data directory: $data_dir"
# Generate config file using edit_config.py
script_args="
    --model $model \
    --litpose_frame $train_frame \
    --seed $seed \
    --litpose_config configs/litpose/config_${dataset}.yaml \
    --output_dir configs/litpose \
    --epochs ${epochs} \
    --mode ${mode} \
    --dataset ${dataset} \
    --data_dir ${data_dir} \
    --model_type ${model_type}
"

echo "Script arguments: $script_args"

generated_config=$(python litpose/edit_config.py $script_args)

# Validate that config was generated
if [ -z "$generated_config" ]; then
    echo "Error: Failed to generate config file"
    exit 1
fi

# Convert to absolute path to ensure it's found regardless of working directory
if [[ "$generated_config" != /* ]]; then
    generated_config="${pwd}/${generated_config}"
fi

echo "Generated config file: $generated_config"

# Set output directory for litpose
litpose_args="
    $generated_config \
    --output_dir outputs/litpose/${dataset}/ds-${dataset}_mode-${mode}_model-${model}_type-${model_type}_frame-${train_frame}_epoch-${epochs}_seed-${seed}
"

echo "Litpose arguments: $litpose_args"

# Run litpose train
litpose train $litpose_args

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

