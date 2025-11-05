#!/bin/bash

#SBATCH --job-name="ibl_encoding"
#SBATCH --output="ibl_encoding.%j.out"
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=8
#SBATCH --gpus-per-task=8
#SBATCH --cpus-per-task=32
#SBATCH --mem=256g
#SBATCH --time=48:00:00
#SBATCH --export=ALL

eid=$1
model=$2
dataset=${3:-ibl-mouse-separate}
start_time=$(date +%s)
echo "Start time: $start_time"
. ~/.bashrc

cd ../..

conda activate mv

# Get the script path
script_path=encoding/ibl_encoding.py

# Get the accelerate command from the setup script
accelerate_cmd=$(./scripts/setup_multi_gpu.sh)
echo "Accelerate command is: $accelerate_cmd"

# Script arguments
script_args="
    --eid ${eid} \
    --data_dir data/encoding/${dataset} \
    --config configs/encoding/${model}_${dataset}.yaml \
    --batch_size 4
"
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

