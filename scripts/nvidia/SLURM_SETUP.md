# SLURM Setup Guide for Single-Node GPU Management

## Overview

This guide helps you install and configure SLURM on your single-node machine with 8 NVIDIA A100 GPUs for better job management and log tracking.

## Why SLURM for Single-Node?

Even though you're the only user, SLURM provides:
- **Job Queue Management**: Automatically queue and run jobs
- **GPU Resource Management**: Prevents conflicts when multiple jobs want GPUs
- **Easy Log Management**: All logs automatically saved with job IDs
- **Job Monitoring**: Easy to see what's running, queued, or completed
- **Job History**: Track all your jobs

## Installation

### Quick Install

```bash
cd /home/nvidia/Projects/multi-view/scripts/nvidia
sudo ./install_slurm.sh
```

### Manual Installation Steps

1. **Install SLURM packages:**
   ```bash
   sudo apt-get update
   sudo apt-get install -y slurm-wlm slurm-wlm-doc slurm-client
   ```

2. **Configure SLURM** (the script does this automatically)
   - Configuration file: `/etc/slurm-llnl/slurm.conf`
   - GPU resources configured for 8 GPUs

3. **Start services:**
   ```bash
   sudo systemctl start munge
   sudo systemctl start slurmctld
   sudo systemctl start slurmd
   sudo systemctl enable munge slurmctld slurmd
   ```

4. **Verify installation:**
   ```bash
   sinfo              # Should show your node
   squeue             # Should show empty (no jobs)
   ```

## Configuration

The installation script creates a basic configuration with:
- **Partition**: `gpu` (default)
- **GPUs**: 8 GPUs available
- **CPUs**: 64 (adjust if needed)
- **Memory**: 512GB (adjust if needed)

To modify settings, edit `/etc/slurm-llnl/slurm.conf` and restart:
```bash
sudo systemctl restart slurmctld slurmd
```

## Using SLURM

### Submit a Job

```bash
cd /home/nvidia/Projects/multi-view/scripts/nvidia
sbatch pretrain.sh mae_ibl-mouse-separate.yaml
```

### Check Job Status

```bash
squeue -u $USER          # Your jobs
squeue                   # All jobs
```

### View Job Details

```bash
scontrol show job <job_id>
```

### Cancel a Job

```bash
scancel <job_id>
scancel -u $USER         # Cancel all your jobs
```

### View Logs

Logs are automatically saved with job ID:
```bash
ls *.out                 # Job output files
cat pretrain.12345.out   # View specific job log
tail -f pretrain.12345.out  # Follow log in real-time
```

### Monitor GPUs

```bash
# In another terminal
nvitop                   # Interactive GPU monitor
```

## Job Submission Examples

### Submit with Specific GPU Count

```bash
# Use 4 GPUs
sbatch --gpus=4 pretrain.sh config.yaml

# Use 1 GPU
sbatch --gpus=1 ibl_encoding.sh eid model
```

### Submit with Resource Limits

```bash
sbatch --gpus=8 --cpus-per-task=32 --mem=256g pretrain.sh config.yaml
```

### Submit and Get Job ID

```bash
job_id=$(sbatch --parsable pretrain.sh config.yaml)
echo "Job submitted: $job_id"
```

## Troubleshooting

### Services Not Starting

```bash
# Check service status
sudo systemctl status slurmctld
sudo systemctl status slurmd

# Check logs
sudo journalctl -u slurmctld -n 50
sudo journalctl -u slurmd -n 50

# Restart services
sudo systemctl restart slurmctld slurmd
```

### Node Not Showing

```bash
# Check node configuration
sudo scontrol show node

# Reconfigure
sudo scontrol reconfigure
```

### GPU Not Available

```bash
# Check GPU configuration
scontrol show node | grep Gres

# Verify nvidia-smi works
nvidia-smi
```

### Permission Issues

```bash
# Ensure you're in the slurm group (if needed)
sudo usermod -a -G slurm $USER
# Log out and back in
```

## Advanced Configuration

### Multiple Partitions

You can create different partitions for different use cases:

```bash
# Edit /etc/slurm-llnl/slurm.conf
# Add:
PartitionName=debug Nodes=$NODE_NAME MaxTime=1:00:00 Default=NO
PartitionName=long Nodes=$NODE_NAME MaxTime=7-00:00:00 Default=YES

# Restart
sudo systemctl restart slurmctld slurmd
```

### Job Priority

SLURM automatically manages priority, but you can set:
```bash
sbatch --priority=1000 script.sh  # Higher priority
```

## Useful Aliases

Add to your `~/.bashrc`:

```bash
alias sq='squeue -u $USER'
alias sqa='squeue'
alias sc='scontrol show job'
alias sj='squeue -u $USER -o "%.10i %.9P %.20j %.8u %.2t %.10M %.6D %R"'
```

## Next Steps

After installation:
1. Test with a simple job: `sbatch --test-only pretrain.sh config.yaml`
2. Submit your first real job
3. Set up log monitoring (consider `tail -f` or a log viewer)
4. Use `nvitop` in another terminal to monitor GPU usage

## Resources

- SLURM Documentation: https://slurm.schedmd.com/documentation.html
- Quick Start: https://slurm.schedmd.com/quickstart.html

