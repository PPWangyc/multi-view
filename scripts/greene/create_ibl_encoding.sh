#!/bin/bash

#SBATCH --account=pr_136_tandon_advanced
#SBATCH --job-name="create_ibl_encoding"
#SBATCH --output="create_ibl_encoding.%j.out"
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem 64G   
#SBATCH --export=ALL
#SBATCH -t 48:00:00
#SBATCH --export=ALL

eid=$1

. ~/.bashrc

cd ../..

conda activate mv

script_path=src/create_dataset/create_ibl_encoding.py
script_args="
--eid $eid
"

# create separate mirror mouse ssl
python $script_path $script_args

conda deactivate

cd scripts/greene
