#!/bin/bash

# Install and configure SLURM for single-node GPU management
# This script sets up SLURM on Ubuntu 22.04 for a single-node machine with 8 GPUs

set -e

echo "=== SLURM Installation for Single-Node GPU Management ==="
echo ""

# Check if running as root or with sudo
if [ "$EUID" -ne 0 ]; then 
    echo "This script requires sudo privileges. Please run with sudo:"
    echo "  sudo $0"
    exit 1
fi

HOSTNAME=$(hostname)
NODE_NAME=$(hostname -s)

# Detect system resources
NUM_CPUS=$(nproc)
TOTAL_MEM_GB=$(free -g | grep Mem | awk '{print $2}')
NUM_GPUS=$(nvidia-smi --query-gpu=count --format=csv,noheader,nounits 2>/dev/null | head -1 || echo "8")

echo "Hostname: $HOSTNAME"
echo "Node name: $NODE_NAME"
echo "Detected resources:"
echo "  CPUs: $NUM_CPUS"
echo "  Memory: ${TOTAL_MEM_GB}GB"
echo "  GPUs: $NUM_GPUS"
echo ""

# Step 1: Install SLURM packages
echo "Step 1: Installing SLURM packages..."
apt-get update
apt-get install -y slurm-wlm slurm-wlm-doc slurm-client

# Step 2: Create SLURM configuration directory
echo ""
echo "Step 2: Creating SLURM configuration..."
mkdir -p /etc/slurm-llnl

# Step 3: Generate SLURM configuration
echo ""
echo "Step 3: Generating SLURM configuration..."
cat > /etc/slurm-llnl/slurm.conf <<EOF
#
# Example slurm.conf file for single-node setup with GPUs
#

# Cluster name
ClusterName=local

# Slurm configuration
SlurmctldHost=$NODE_NAME
SlurmctldPort=6817
SlurmdPort=6818
AuthType=auth/munge
StateSaveLocation=/var/spool/slurm-llnl
SlurmdSpoolDir=/var/spool/slurm-llnl/slurmd
SwitchType=switch/none
MpiDefault=none
SlurmctldPidFile=/var/run/slurmctld.pid
SlurmdPidFile=/var/run/slurmd.pid
SlurmctldLogFile=/var/log/slurm-llnl/slurmctld.log
SlurmdLogFile=/var/log/slurm-llnl/slurmd.log

# Scheduling
SchedulerType=sched/backfill
SelectType=select/cons_tres
SelectTypeParameters=CR_CPU_Memory

# Accounting (optional, for single user)
AccountingStorageType=accounting_storage/none

# Job priority
PriorityType=priority/multifactor
PriorityDecayHalfLife=7-0
PriorityMaxAge=30-0
PriorityFavorSmall=NO
PriorityWeightAge=1000
PriorityWeightFairshare=0
PriorityWeightJobSize=1000
PriorityWeightPartition=10000
PriorityWeightQOS=0

# Node configuration
NodeName=$NODE_NAME NodeAddr=localhost CPUs=$NUM_CPUS RealMemory=$((TOTAL_MEM_GB * 1024)) Sockets=1 CoresPerSocket=$NUM_CPUS ThreadsPerCore=1 State=UNKNOWN

# Partition configuration - GPU partition
PartitionName=gpu Nodes=$NODE_NAME Default=YES MaxTime=INFINITE MaxNodes=1 AllowGroups=ALL AllowAccounts=ALL

# GPU configuration
GresTypes=gpu
NodeName=$NODE_NAME Gres=gpu:$NUM_GPUS

# Prolog and Epilog (optional)
#Prolog=/etc/slurm-llnl/prolog.sh
#Epilog=/etc/slurm-llnl/epilog.sh
EOF

# Step 4: Create cgroup configuration (optional but recommended)
echo ""
echo "Step 4: Creating cgroup configuration..."
cat > /etc/slurm-llnl/cgroup.conf <<EOF
CgroupAutomount=yes
ConstrainCores=yes
ConstrainRAMSpace=yes
ConstrainDevices=yes
EOF

# Step 4.5: Create GPU resource configuration
echo ""
echo "Step 4.5: Creating GPU resource configuration..."
cat > /etc/slurm-llnl/gres.conf <<EOF
# GPU resource configuration
Name=gpu Type=unknown File=/dev/nvidia0
Name=gpu Type=unknown File=/dev/nvidia1
Name=gpu Type=unknown File=/dev/nvidia2
Name=gpu Type=unknown File=/dev/nvidia3
Name=gpu Type=unknown File=/dev/nvidia4
Name=gpu Type=unknown File=/dev/nvidia5
Name=gpu Type=unknown File=/dev/nvidia6
Name=gpu Type=unknown File=/dev/nvidia7
EOF
ln -sf /etc/slurm-llnl/gres.conf /etc/slurm/gres.conf 2>/dev/null || true

# Step 5: Create log directory
echo ""
echo "Step 5: Creating log directory..."
mkdir -p /var/log/slurm-llnl
chown -R slurm:slurm /var/log/slurm-llnl

# Step 6: Create spool directory
echo ""
echo "Step 6: Creating spool directory..."
mkdir -p /var/spool/slurm-llnl
chown -R slurm:slurm /var/spool/slurm-llnl

# Step 6.5: Create symlinks for systemd (systemd expects /etc/slurm/)
echo ""
echo "Step 6.5: Creating symlinks for systemd..."
mkdir -p /etc/slurm
ln -sf /etc/slurm-llnl/slurm.conf /etc/slurm/slurm.conf
ln -sf /etc/slurm-llnl/cgroup.conf /etc/slurm/cgroup.conf 2>/dev/null || true

# Step 7: Initialize SLURM database (munged)
echo ""
echo "Step 7: Setting up munge (authentication)..."
if [ ! -f /etc/munge/munge.key ]; then
    dd if=/dev/urandom bs=1 count=1024 > /etc/munge/munge.key
    chmod 400 /etc/munge/munge.key
    chown munge:munge /etc/munge/munge.key
fi

# Step 8: Start services
echo ""
echo "Step 8: Starting SLURM services..."
systemctl enable munge
systemctl start munge

systemctl enable slurmctld
systemctl enable slurmd

# Step 9: Initialize SLURM
echo ""
echo "Step 9: Initializing SLURM..."
systemctl restart slurmctld
systemctl restart slurmd

# Wait a moment for services to start
sleep 2

# Step 10: Verify installation
echo ""
echo "Step 10: Verifying installation..."
if systemctl is-active --quiet slurmctld && systemctl is-active --quiet slurmd; then
    echo "✓ SLURM services are running"
    
    # Wait a moment for node registration
    sleep 2
    
    # Check node status
    echo ""
    echo "Node status:"
    sinfo -N -l 2>/dev/null || echo "Note: sinfo may need a moment to show nodes"
    
    # If node is in DRAIN state due to GPU detection, manually set to IDLE
    echo ""
    echo "Step 10.5: Configuring node state..."
    node_state=$(sinfo -N -h -o "%T" 2>/dev/null | head -1)
    if [[ "$node_state" == *"drain"* ]] || [[ "$node_state" == *"DRAIN"* ]]; then
        echo "  Node is in DRAIN state, setting to IDLE..."
        scontrol update nodename=$NODE_NAME state=idle reason="GPU configured" 2>/dev/null || true
        sleep 2
    fi
    
    echo ""
    echo "Final node status:"
    sinfo -N -l 2>/dev/null || true
    
    echo ""
    echo "=== SLURM Installation Complete ==="
    echo ""
    echo "Useful commands:"
    echo "  squeue -u \$USER          # View your jobs"
    echo "  sinfo                     # View node/partition status"
    echo "  sbatch script.sh          # Submit a job"
    echo "  scancel <job_id>          # Cancel a job"
    echo "  scontrol show node        # Show node details"
    echo ""
    echo "If node shows as DRAINED, run:"
    echo "  sudo scontrol update nodename=$NODE_NAME state=idle"
    echo ""
    echo "Test with: sbatch --test-only /path/to/your/script.sh"
else
    echo "✗ SLURM services failed to start"
    echo "Check logs:"
    echo "  journalctl -u slurmctld -n 50"
    echo "  journalctl -u slurmd -n 50"
    exit 1
fi

