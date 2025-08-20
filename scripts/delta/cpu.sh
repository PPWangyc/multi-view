#!/bin/bash
#SBATCH -J Jupyter-CPU
#SBATCH --output=./cpu_%j.out
#SBATCH --account=beez-delta-cpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=cpu
#SBATCH --time=04:30:00
#SBATCH --mem=64g
#SBATCH --cpus-per-task=8
srun jupyter-notebook --no-browser --ip=0.0.0.0