#!/bin/bash

#SBATCH --job-name="setup_env"
#SBATCH --output="setup_env.%j.out"
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64g
#SBATCH --gpus=1
#SBATCH --time=2:00:00
#SBATCH --export=ALL

cd ../..

pwd=$(pwd)

echo "Setting up environment in $pwd..."

. ~/.bashrc

# Accept conda Terms of Service for required channels
echo "Accepting conda Terms of Service..."
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main || true
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r || true

conda env create -f env.yaml

. ~/.bashrc

conda activate mv

cd ..

git clone git@github.com:PPWangyc/lightning-pose.git

cd lightning-pose

pip install -e .

cd ..

cd $pwd/scripts/nvidia

conda deactivate