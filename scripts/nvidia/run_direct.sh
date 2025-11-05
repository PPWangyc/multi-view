#!/bin/bash

# Run a job directly without SLURM
# Usage: ./run_direct.sh <script_name> [args...]
# Example: ./run_direct.sh pretrain.sh mae_ibl-mouse-separate.yaml

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

# Remove SLURM directives from the script and run it
# This strips out #SBATCH lines and runs the rest
echo "Running $script_name directly (without SLURM)..."
grep -v "^#SBATCH" "$script_name" | bash -s "$@"

