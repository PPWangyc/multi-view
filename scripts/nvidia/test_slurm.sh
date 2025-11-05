#!/bin/bash

# Test SLURM installation and configuration

echo "=== SLURM Test Script ==="
echo ""

# Test 1: Check if SLURM commands are available
echo "1. Checking SLURM commands..."
commands=("sbatch" "squeue" "sinfo" "scontrol" "scancel")
all_ok=true
for cmd in "${commands[@]}"; do
    if command -v $cmd &> /dev/null; then
        echo "  ✓ $cmd is available"
    else
        echo "  ✗ $cmd is NOT available"
        all_ok=false
    fi
done

if [ "$all_ok" = false ]; then
    echo ""
    echo "SLURM is not installed. Run: sudo ./install_slurm.sh"
    exit 1
fi

echo ""

# Test 2: Check SLURM services
echo "2. Checking SLURM services..."
if systemctl is-active --quiet slurmctld 2>/dev/null; then
    echo "  ✓ slurmctld is running"
else
    echo "  ✗ slurmctld is NOT running"
    echo "    Start with: sudo systemctl start slurmctld"
fi

if systemctl is-active --quiet slurmd 2>/dev/null; then
    echo "  ✓ slurmd is running"
else
    echo "  ✗ slurmd is NOT running"
    echo "    Start with: sudo systemctl start slurmd"
fi

echo ""

# Test 3: Check node status
echo "3. Checking node status..."
if sinfo -N -l 2>/dev/null | grep -q "UP"; then
    echo "  ✓ Node is UP"
    sinfo -N -l
else
    echo "  ✗ Node status unknown"
    echo "    Run: sinfo -N -l"
fi

echo ""

# Test 4: Check GPU configuration
echo "4. Checking GPU configuration..."
if scontrol show node | grep -q "Gres=gpu"; then
    echo "  ✓ GPUs configured in SLURM"
    scontrol show node | grep Gres
else
    echo "  ✗ GPUs not configured"
    echo "    Check /etc/slurm-llnl/slurm.conf"
fi

echo ""

# Test 5: Check partition
echo "5. Checking partitions..."
sinfo -o "%.15P %.10a %.10l %.6D %.6t %.8z %.6m %.8d %.6w %.8f %.20G"
echo ""

# Test 6: Test job submission (dry run)
echo "6. Testing job submission (dry run)..."
cat > /tmp/test_slurm_job.sh <<'TESTEOF'
#!/bin/bash
#SBATCH --job-name=test
#SBATCH --output=test.%j.out
#SBATCH --time=00:01:00
#SBATCH --gpus=1
echo "Test job running"
TESTEOF

chmod +x /tmp/test_slurm_job.sh

if sbatch --test-only /tmp/test_slurm_job.sh 2>&1 | grep -q "test"; then
    echo "  ✓ Job submission test passed"
else
    echo "  ⚠ Job submission test had warnings (this may be OK)"
    sbatch --test-only /tmp/test_slurm_job.sh
fi

rm -f /tmp/test_slurm_job.sh

echo ""
echo "=== Test Complete ==="
echo ""
echo "If all tests pass, you can start submitting jobs:"
echo "  sbatch pretrain.sh config.yaml"

