#!/bin/bash

#SBATCH --account=pr_136_tandon_advanced
#SBATCH --job-name="ibl_encoding"
#SBATCH --output="ibl_encoding.%j.out"
#SBATCH -N 4
#SBATCH -c 16
#SBATCH --ntasks-per-node=1
#SBATCH --mem=64g
#SBATCH --gres=gpu:1
#SBATCH -t 48:00:00
#SBATCH --constraint=h100|a100
#SBATCH --export=ALL
eid=$1
config_file=$2
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
    --data_dir data/encoding/ibl-mouse-separate
"
$accelerate_cmd $script_path $script_args
cd scripts/delta
conda activate mv

end_time=$(date +%s)
echo "End time: $end_time"
time_taken=$((end_time - start_time))
# show time taken in hours, minutes, and seconds
hours=$((time_taken / 3600))
minutes=$(((time_taken % 3600) / 60))
seconds=$((time_taken % 60))
echo "Time taken: ${hours}:$(printf "%02d" $minutes):$(printf "%02d" $seconds)"