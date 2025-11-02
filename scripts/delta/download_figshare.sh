#!/bin/bash

#SBATCH -J download_figshare
#SBATCH --output=./download_figshare_%j.out
#SBATCH --account=bdeu-delta-cpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=cpu
#SBATCH --time=48:00:00
#SBATCH --mem=64g
#SBATCH --cpus-per-task=16

url=$1
output_dir=$2

. ~/.bashrc

cd ../..

conda activate mv

script_path=scripts/download_figshare.py

python $script_path --url $url --output-dir $output_dir

conda deactivate

cd scripts/delta
