#!/bin/bash

# Script to submit multiple litpose jobs with different seeds
# Usage: ./run_litpose.sh <dataset> <model> <model_type> <train_frame>

if [ $# -lt 4 ]; then
    echo "Usage: $0 <dataset> <model> <model_type> <train_frame>"
    echo "Example: $0 fly-anipose vitb_dinov3 mv 100"
    exit 1
fi

dataset=$1
model=$2
model_type=$3
train_frame=$4

# Loop through seeds [0, 1, 2]
for seed in 0 1 2; do
    echo "Submitting job for seed=$seed"
    if [ "$model" = "vggt" ] || [[ "$model" == *"vitl"* ]]; then
        sbatch --gpus=2 --gpus-per-task=2 --cpus-per-task=64 --mem=256g litpose.sh $dataset $model $model_type $train_frame $seed
    else
        sbatch litpose.sh $dataset $model $model_type $train_frame $seed
    fi
done

echo "All jobs submitted successfully!"

