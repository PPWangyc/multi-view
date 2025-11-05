# Quick Start: SLURM for GPU Job Management

## Installation

```bash
cd /home/nvidia/Projects/multi-view/scripts/nvidia
sudo ./install_slurm.sh
```

This will:
- Install SLURM packages
- Auto-detect your system resources (255 CPUs, 2015GB RAM, 8 GPUs)
- Configure SLURM for single-node GPU management
- Start SLURM services

## Verify Installation

```bash
./test_slurm.sh
```

## Submit Your First Job

```bash
# Submit a pretraining job
sbatch pretrain.sh mae_ibl-mouse-separate.yaml

# Check job status
squeue -u $USER

# View logs (replace 12345 with your job ID)
cat pretrain.12345.out
tail -f pretrain.12345.out  # Follow log in real-time
```

## Common Commands

```bash
# Submit jobs
sbatch pretrain.sh config.yaml
sbatch ibl_encoding.sh eid model
sbatch litpose.sh config.yaml

# Check status
squeue -u $USER          # Your jobs
squeue                    # All jobs
sinfo                     # Node/partition status

# Cancel jobs
scancel <job_id>          # Cancel specific job
scancel -u $USER         # Cancel all your jobs

# View job details
scontrol show job <job_id>

# Monitor GPUs (in another terminal)
nvitop
```

## Log Management

All job logs are automatically saved with job IDs:
- `pretrain.12345.out` - Job output
- `pretrain.12345.err` - Job errors (if any)

List all logs:
```bash
ls -lt *.out | head      # Most recent logs
```

## Tips

1. **Use aliases** (add to `~/.bashrc`):
   ```bash
   alias sq='squeue -u $USER'
   alias sj='sbatch'
   alias sc='scontrol show job'
   ```

2. **Monitor multiple jobs**:
   ```bash
   watch -n 5 'squeue -u $USER'
   ```

3. **Check GPU usage**:
   ```bash
   nvitop  # Interactive
   gpustat -i 1  # Lightweight
   ```

## Troubleshooting

If services don't start:
```bash
sudo systemctl status slurmctld slurmd
sudo journalctl -u slurmctld -n 50
sudo systemctl restart slurmctld slurmd
```

For detailed setup guide, see `SLURM_SETUP.md`

