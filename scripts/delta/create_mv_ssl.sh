#!/bin/bash
#SBATCH -J create_mv_ssl
#SBATCH --output=./create_mv_ssl_%j.out
#SBATCH --account=beez-delta-cpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=cpu
#SBATCH --time=12:30:00
#SBATCH --mem=64g
#SBATCH --cpus-per-task=8

dataset=$1

. ~/.bashrc

cd ../..

conda activate mv

script_path=src/create_dataset/create_mv_ssl.py
script_args="
--input_dir data
--output_dir data/ssl
--dataset $dataset
"

# create separate mirror mouse ssl
python $script_path $script_args

conda deactivate

cd scripts/delta
