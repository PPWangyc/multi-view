#!/bin/bash

memory=64g
constraint_a100_h100="a100|h100"
constraint_rtx8000="rtx8000"

config_file=$1

# Function to submit job for specific GPU type
submit_job() {
    local gpu_type=$1
    local num_nodes=$2
    local constraint=$3
    
    multi_node_args="
    --cpus-per-task=16
    --gres=gpu:1
    --constraint=$constraint
    --ntasks=$num_nodes
    --nodes=$num_nodes
    --mem=$memory
    --time=48:00:00
    --output=pretrain_${gpu_type}.%j.out
    "
    
    script_path=pretrain.sh
    
    sbatch $multi_node_args $script_path $config_file
}

# Submit job for A100/H100 (4 nodes)
echo "Submitting job for A100/H100 (4 nodes)..."
job_a100_h100=$(submit_job "a100_h100" 4 "$constraint_a100_h100" | grep -o '[0-9]*')

# Submit job for RTX8000 (2 nodes)
echo "Submitting job for RTX8000 (8 nodes)..."
job_rtx8000=$(submit_job "rtx8000" 8 "$constraint_rtx8000" | grep -o '[0-9]*')

echo "Submitted jobs:"
echo "  A100/H100 job ID: $job_a100_h100"
echo "  RTX8000 job ID: $job_rtx8000"

# Create a monitoring script to cancel the other job when one starts
monitor_script="monitor_jobs_$$.sh"
cat > $monitor_script << EOF
#!/bin/bash
# Monitor script to cancel one job when the other starts

job_a100_h100=$job_a100_h100
job_rtx8000=$job_rtx8000

echo "Monitoring jobs for mutual cancellation..."

while true; do
    # Check if A100/H100 job is running
    if squeue -j \$job_a100_h100 -h -o "%T" | grep -q "RUNNING"; then
        echo "A100/H100 job is running. Canceling RTX8000 job..."
        scancel \$job_rtx8000
        break
    fi
    
    # Check if RTX8000 job is running
    if squeue -j \$job_rtx8000 -h -o "%T" | grep -q "RUNNING"; then
        echo "RTX8000 job is running. Canceling A100/H100 job..."
        scancel \$job_a100_h100
        break
    fi
    
    # Check if either job has completed or failed
    if ! squeue -j \$job_a100_h100,\$job_rtx8000 >/dev/null 2>&1; then
        echo "Both jobs have completed or failed. Stopping monitor."
        break
    fi
    
    sleep 30  # Check every 30 seconds
done

# Clean up monitor script
rm -f $monitor_script
EOF

chmod +x $monitor_script

# Start the monitoring script in background
echo "Starting job monitor..."
nohup $monitor_script > monitor_$$.log 2>&1 &

echo "Job submission complete. Monitor script is running in background."
echo "Monitor log: monitor_$$.log"