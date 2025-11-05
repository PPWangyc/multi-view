#!/bin/bash

# Generic job submission wrapper
# Usage: ./submit_job.sh <script_name> [args...]
# Example: ./submit_job.sh pretrain.sh mae_ibl-mouse-separate.yaml

if [ $# -lt 1 ]; then
    echo "Usage: $0 <script_name> [args...]"
    echo "Example: $0 pretrain.sh mae_ibl-mouse-separate.yaml"
    exit 1
fi

script_name=$1
shift  # Remove first argument, rest are passed to the script

if [ ! -f "$script_name" ]; then
    echo "Error: Script '$script_name' not found"
    exit 1
fi

# Check if SLURM is available
if command -v sbatch &> /dev/null; then
    echo "Submitting job via SLURM..."
    job_id=$(sbatch "$script_name" "$@" | grep -o '[0-9]*')
    echo "Job submitted with ID: $job_id"
    echo "Monitor with: squeue -u $USER"
    echo "Cancel with: scancel $job_id"
else
    echo "SLURM not available, running directly..."
    echo "Warning: This will run directly on the current terminal"
    bash "$script_name" "$@"
fi

