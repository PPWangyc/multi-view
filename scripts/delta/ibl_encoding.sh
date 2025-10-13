#!/bin/bash

#SBATCH --account=beez-delta-gpu
#SBATCH --partition=gpuA40x4, gpuA40x4-preempt
#SBATCH --job-name="ibl-encoding"
#SBATCH --output="ibl-encoding.%j.out"
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=98g   
#SBATCH -t 2-00
#SBATCH --export=ALL

eid=$1
model=$2 # mvmae / svmae / mae / ijepa / svijepa
avail_views=$3 # left+right

dataset=ibl-mouse-separate
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
    --data_dir data/encoding/ibl-mouse-separate \
    --config configs/encoding/${model}_${dataset}.yaml \
    --avail_views ${avail_views}
"
echo "Script arguments are: $script_args"
$accelerate_cmd $script_path $script_args
cd scripts/delta
conda deactivate

end_time=$(date +%s)
echo "End time: $end_time"
time_taken=$((end_time - start_time))
# show time taken in hours, minutes, and seconds
hours=$((time_taken / 3600))
minutes=$(((time_taken % 3600) / 60))
seconds=$((time_taken % 60))
echo "Time taken: ${hours}:$(printf "%02d" $minutes):$(printf "%02d" $seconds)"