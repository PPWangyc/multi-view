#!/bin/bash

# Install GPU monitoring and management tools

echo "Installing GPU monitoring tools..."

# Install nvitop (best interactive GPU monitor)
if ! command -v nvitop &> /dev/null; then
    echo "Installing nvitop..."
    pip install nvitop
else
    echo "nvitop already installed"
fi

# Install gpustat (lightweight alternative)
if ! command -v gpustat &> /dev/null; then
    echo "Installing gpustat..."
    pip install gpustat
else
    echo "gpustat already installed"
fi

# Check if SLURM is installed
if command -v sbatch &> /dev/null; then
    echo "SLURM is already installed"
else
    echo "SLURM is not installed"
    echo "To install SLURM for single-node setup, see README.md"
    echo "Or use the scripts without SLURM (they will run directly)"
fi

echo ""
echo "Installation complete!"
echo ""
echo "Usage examples:"
echo "  nvitop                    # Interactive GPU monitor"
echo "  gpustat -i 1             # Lightweight GPU monitor"
echo "  watch -n 1 nvidia-smi   # Basic nvidia-smi monitoring"

