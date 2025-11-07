#!/bin/bash

# Check for available GPUs and return device IDs and count
# Handles both SLURM and non-SLURM environments
# Returns: num_processes, device_ids (comma-separated)
# Also sets CUDA_VISIBLE_DEVICES if not already set

# Check if running under SLURM
if [ -n "$SLURM_JOB_ID" ] || [ -n "$SLURM_JOB_NODELIST" ]; then
    # Running under SLURM - use SLURM-allocated GPUs
    
    # First, check if SLURM already set CUDA_VISIBLE_DEVICES
    if [ -n "$CUDA_VISIBLE_DEVICES" ]; then
        # SLURM set CUDA_VISIBLE_DEVICES - use it
        num_processes=$(python -c "import os; devs = os.environ.get('CUDA_VISIBLE_DEVICES', ''); print(len([d for d in devs.split(',') if d.strip()]) if devs else 0)")
        available_devices="$CUDA_VISIBLE_DEVICES"
        echo "Using $num_processes GPU(s) from SLURM CUDA_VISIBLE_DEVICES: $available_devices" >&2
    else
        # SLURM didn't set CUDA_VISIBLE_DEVICES, try to get from SLURM environment variables
        if [ -n "$SLURM_STEP_GPUS" ]; then
            # Use SLURM_STEP_GPUS if available
            available_devices="$SLURM_STEP_GPUS"
            num_processes=$(python -c "devs = '$SLURM_STEP_GPUS'; print(len([d for d in devs.split(',') if d.strip()]) if devs else 0)")
            export CUDA_VISIBLE_DEVICES="$available_devices"
            echo "Using $num_processes GPU(s) from SLURM_STEP_GPUS: $available_devices" >&2
        elif [ -n "$SLURM_JOB_GPUS" ]; then
            # Use SLURM_JOB_GPUS if available
            available_devices="$SLURM_JOB_GPUS"
            num_processes=$(python -c "devs = '$SLURM_JOB_GPUS'; print(len([d for d in devs.split(',') if d.strip()]) if devs else 0)")
            export CUDA_VISIBLE_DEVICES="$available_devices"
            echo "Using $num_processes GPU(s) from SLURM_JOB_GPUS: $available_devices" >&2
        else
            # Try to query SLURM for allocated GPUs using scontrol
            if command -v scontrol &> /dev/null && [ -n "$SLURM_JOB_ID" ]; then
                # Get GPU allocation from scontrol
                gpu_allocation=$(scontrol show job "$SLURM_JOB_ID" 2>/dev/null | grep -i "gres/gpu" | head -1)
                if [ -n "$gpu_allocation" ]; then
                    # Extract GPU count from gres/gpu
                    gpu_count=$(echo "$gpu_allocation" | grep -oP 'gres/gpu:\K\d+' | head -1)
                    if [ -n "$gpu_count" ] && [ "$gpu_count" -gt 0 ]; then
                        # Generate device list (0 to gpu_count-1)
                        # Note: SLURM remaps allocated GPUs to 0,1,2,... so we use consecutive IDs
                        available_devices=$(seq -s, 0 $((gpu_count - 1)))
                        num_processes=$gpu_count
                        export CUDA_VISIBLE_DEVICES="$available_devices"
                        echo "Using $num_processes GPU(s) from SLURM allocation: $available_devices" >&2
                    else
                        echo "ERROR: Could not determine GPU count from SLURM." >&2
                        echo "ERROR: Running under SLURM but cannot determine GPU allocation." >&2
                        echo "ERROR: Please check SLURM configuration or set CUDA_VISIBLE_DEVICES manually." >&2
                        exit 1
                    fi
                else
                    echo "ERROR: Could not query SLURM for GPU allocation." >&2
                    echo "ERROR: Running under SLURM but cannot determine GPU allocation." >&2
                    echo "ERROR: Please check SLURM configuration or set CUDA_VISIBLE_DEVICES manually." >&2
                    exit 1
                fi
            else
                echo "ERROR: Running under SLURM but cannot determine GPU allocation." >&2
                echo "ERROR: SLURM environment variables not set properly." >&2
                echo "ERROR: Please check SLURM configuration or set CUDA_VISIBLE_DEVICES manually." >&2
                exit 1
            fi
        fi
    fi
else
    # Not running under SLURM - check available GPUs on the system
    
    # Check if nvidia-smi is available
    if ! command -v nvidia-smi &> /dev/null; then
        echo "WARNING: nvidia-smi not found. Assuming 1 process with CPU." >&2
        echo "1"
        echo ""
        exit 0
    fi
    
    # Get total number of GPUs
    total_gpus=$(nvidia-smi --query-gpu=count --format=csv,noheader,nounits 2>/dev/null | head -1)
    
    if [ -z "$total_gpus" ] || [ "$total_gpus" -eq 0 ]; then
        echo "WARNING: No GPUs detected. Falling back to 1 process (CPU)." >&2
        echo "1"
        echo ""
        exit 0
    fi
    
    # Find available GPUs by checking for active processes
    available_gpus=()
    for i in $(seq 0 $((total_gpus - 1))); do
        # Check if any processes are using this GPU
        # Query for compute processes (PIDs) on this GPU
        processes=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader -i $i 2>/dev/null | grep -v "^$" | wc -l)
        
        # Also check GPU utilization and memory as a secondary check
        gpu_info=$(nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits -i $i 2>/dev/null)
        
        is_available=true
        
        # If there are processes, GPU is not available
        if [ "$processes" -gt 0 ]; then
            is_available=false
        elif [ -n "$gpu_info" ]; then
            # Parse the output: utilization.gpu,memory.used,memory.total
            utilization=$(echo "$gpu_info" | cut -d',' -f1 | tr -d ' ')
            mem_used=$(echo "$gpu_info" | cut -d',' -f2 | tr -d ' ')
            mem_total=$(echo "$gpu_info" | cut -d',' -f3 | tr -d ' ')
            
            # Consider GPU available if utilization and memory usage are low
            if [ -n "$utilization" ] && [ -n "$mem_used" ] && [ -n "$mem_total" ] && [ "$mem_total" -gt 0 ]; then
                mem_percent=$((mem_used * 100 / mem_total))
                
                # GPU is considered in use if utilization > 5% OR memory > 10%
                if [ "$utilization" -ge 5 ] || [ "$mem_percent" -ge 10 ]; then
                    is_available=false
                fi
            fi
        fi
        
        if [ "$is_available" = true ]; then
            available_gpus+=($i)
        fi
    done
    
    # Handle case where no GPUs are available
    if [ ${#available_gpus[@]} -eq 0 ]; then
        echo "WARNING: No available GPUs found. All GPUs appear to be in use." >&2
        echo "WARNING: Falling back to 1 process (CPU)." >&2
        echo "1"
        echo ""
        exit 0
    fi
    
    # Output number of processes (number of available GPUs)
    num_processes=${#available_gpus[@]}
    
    # Output device IDs as comma-separated string
    available_devices=$(IFS=','; echo "${available_gpus[*]}")
    
    # Set CUDA_VISIBLE_DEVICES to available GPUs
    export CUDA_VISIBLE_DEVICES="$available_devices"
    echo "Using $num_processes GPU(s): $available_devices" >&2
fi

# Output results (stdout for capturing)
echo "$num_processes"
echo "$available_devices"
