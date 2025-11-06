#!/bin/bash

#SBATCH --job-name="create_mv_ssl"
#SBATCH --output="create_mv_ssl.%j.out"
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=128g
#SBATCH --time=48:00:00
#SBATCH --export=ALL

dataset=$1

. ~/.bashrc

source "$HOME/miniconda3/etc/profile.d/conda.sh"

cd ../..

conda activate mv

# Set PYTHONPATH to include src directory so utils can be imported
export PYTHONPATH="${PWD}/src:${PYTHONPATH}"

# create separate mirror mouse ssl
python src/create_dataset/create_mv_ssl.py \
    --input_dir data \
    --output_dir data/ssl \
    --dataset $dataset

conda deactivate

cd scripts/nvidia
