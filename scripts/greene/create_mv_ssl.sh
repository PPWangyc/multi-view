#!/bin/bash

#SBATCH --account=pr_136_tandon_advanced
#SBATCH --job-name="create_mv_ssl"
#SBATCH --output="create_mv_ssl.%j.out"
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem 64G   
#SBATCH --export=ALL
#SBATCH -t 18:00:00
#SBATCH --export=ALL

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

cd scripts/greene
