#!/bin/bash
#SBATCH -J Jupyter-GPU
#SBATCH --output=./gpu_%j.out
#SBATCH --account=beez-delta-gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=gpuA40x4   # <-or one of: gpuA100x4 gpuA40x4 gpuA100x8 gpuMI100x8
#SBATCH --time=03:30:00
#SBATCH --mem=64g
#SBATCH --cpus-per-task=8
# # ### GPU options ###
#SBATCH --gpus-per-node=1
#SBATCH --gpus-per-task=1

node_ip=$(srun hostname -I)
echo "Node IP: $node_ip"

srun jupyter-notebook --no-browser --ip=0.0.0.0