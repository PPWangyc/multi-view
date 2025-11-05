# GPU Management Tools for NVIDIA Machine (8 GPUs)

## SLURM Installation (Recommended)

**SLURM is perfect for managing many jobs and logs!**

### Quick Install
```bash
cd /home/nvidia/Projects/multi-view/scripts/nvidia
sudo ./install_slurm.sh
```

This automatically:
- Detects your system resources (255 CPUs, 2015GB RAM, 8 GPUs)
- Configures SLURM for single-node GPU management
- Sets up job queuing and log management

### Verify Installation
```bash
./test_slurm.sh
```

### Quick Start
See `QUICK_START.md` for common commands and usage examples.

For detailed setup guide, see `SLURM_SETUP.md`.

---

## Recommended Tools

### 1. **SLURM** (Recommended for Job Scheduling)
Even for a single-node machine, SLURM provides:
- Job queuing and management
- Resource allocation (prevents GPU conflicts)
- Easy job monitoring (`squeue`, `scontrol`)
- Automatic log management (all logs saved with job IDs)
- Job history and accounting

**Installation:**
```bash
# Use the automated installer
cd /home/nvidia/Projects/multi-view/scripts/nvidia
sudo ./install_slurm.sh
```

**Alternative: SimpleSLURM** (easier single-node setup):
```bash
pip install simpleslurm
```

### 2. **GPU Monitoring Tools**

#### **nvitop** (Recommended - Best UI)
```bash
pip install nvitop
# Usage: nvitop
```
- Interactive TUI with process monitoring
- Shows GPU utilization, memory, temperature
- Can kill processes directly from the UI

#### **gpustat** (Lightweight)
```bash
pip install gpustat
# Usage: gpustat -i 1  # Refresh every 1 second
watch -n 1 gpustat     # Continuous monitoring
```

#### **nvidia-smi** (Built-in)
```bash
watch -n 1 nvidia-smi  # Basic monitoring
```

### 3. **Job Submission Scripts**
See the example scripts in this directory:
- `pretrain.sh` - Multi-GPU pretraining jobs
- `ibl_encoding.sh` - Encoding jobs
- `submit_job.sh` - Generic job submission wrapper

## Quick Start

### Without SLURM (Direct Execution)
```bash
# Run directly with all 8 GPUs
cd /home/nvidia/Projects/multi-view
source scripts/setup_multi_gpu.sh
accelerate_cmd=$(./scripts/setup_multi_gpu.sh)
$accelerate_cmd src/pretrain.py --config configs/pretrain/your_config.yaml
```

### With SLURM
```bash
# Submit a job
cd scripts/nvidia
sbatch pretrain.sh your_config.yaml

# Check job status
squeue -u $USER

# Monitor GPU usage
nvitop
```

## GPU Allocation Examples

### Single GPU
```bash
CUDA_VISIBLE_DEVICES=0 python your_script.py
```

### Multiple GPUs (e.g., 4 GPUs)
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python your_script.py
```

### All 8 GPUs
```bash
# Use accelerate (recommended)
accelerate launch --config_file configs/accelerate/default.yaml --num_processes 8 your_script.py
```

## Useful Commands

```bash
# Check GPU status
nvidia-smi

# Monitor continuously
nvitop

# List all jobs
squeue -u $USER

# Cancel a job
scancel <job_id>

# Check job details
scontrol show job <job_id>
```

