#!/bin/bash

# Check for available GPUs and return device IDs and count
# Returns: num_processes, device_ids (comma-separated)

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
device_ids=$(IFS=','; echo "${available_gpus[*]}")

# Output results (stdout for capturing)
echo "$num_processes"
echo "$device_ids"

