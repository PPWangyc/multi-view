#!/bin/bash

# Check GPU setup and available management tools

echo "=== GPU Management Tools Check ==="
echo ""

# Check GPU count
num_gpus=$(nvidia-smi --query-gpu=gpu_name --format=csv,noheader 2>/dev/null | wc -l)
if [ $? -eq 0 ] && [ $num_gpus -gt 0 ]; then
    echo "✓ GPUs detected: $num_gpus"
    echo "  GPU Details:"
    nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader | sed 's/^/    /'
else
    echo "✗ No GPUs detected or nvidia-smi not available"
fi
echo ""

# Check SLURM
if command -v sbatch &> /dev/null; then
    echo "✓ SLURM is installed"
    echo "  Version: $(sbatch --version 2>&1 | head -n 1)"
    echo "  Queue: $(squeue -u $USER 2>/dev/null | wc -l) jobs in queue"
else
    echo "✗ SLURM is not installed"
    echo "  You can still run jobs directly (see run_direct.sh)"
fi
echo ""

# Check monitoring tools
echo "Monitoring tools:"
if command -v nvitop &> /dev/null; then
    echo "  ✓ nvitop (recommended)"
else
    echo "  ✗ nvitop - Install with: pip install nvitop"
fi

if command -v gpustat &> /dev/null; then
    echo "  ✓ gpustat"
else
    echo "  ✗ gpustat - Install with: pip install gpustat"
fi

if command -v nvidia-smi &> /dev/null; then
    echo "  ✓ nvidia-smi (built-in)"
fi
echo ""

# Check Python tools
echo "Python tools:"
if python -c "import accelerate" 2>/dev/null; then
    echo "  ✓ accelerate"
else
    echo "  ✗ accelerate - Install with: pip install accelerate"
fi

if python -c "import torch" 2>/dev/null; then
    echo "  ✓ PyTorch"
    python -c "import torch; print(f'    CUDA available: {torch.cuda.is_available()}, GPUs: {torch.cuda.device_count()}')"
else
    echo "  ✗ PyTorch"
fi
echo ""

echo "=== Quick Start ==="
echo ""
echo "1. Install tools: ./install_gpu_tools.sh"
echo "2. Monitor GPUs: nvitop"
echo "3. Submit job: ./submit_job.sh pretrain.sh <config_file>"
echo "4. Run directly: ./run_direct.sh pretrain.sh <config_file>"
echo ""

