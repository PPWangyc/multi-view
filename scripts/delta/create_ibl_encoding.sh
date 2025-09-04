#!/bin/bash

#SBATCH -J create_ibl_encoding
#SBATCH --output=./create_ibl_encoding_%j.out
#SBATCH --account=bdeu-delta-cpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=cpu
#SBATCH --time=48:00:00
#SBATCH --mem=64g
#SBATCH --cpus-per-task=16

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
