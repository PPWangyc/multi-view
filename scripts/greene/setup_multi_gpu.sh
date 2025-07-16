#!/bin/bash

# Find a free port
PORT=$(python -c "import socket; s=socket.socket(); s.bind(('',0)); print(s.getsockname()[1]); s.close()")
ip_addr=$(hostname -i)

# Check if running under SLURM
if [ -n "$SLURM_JOB_NODELIST" ]; then
    # Running under SLURM
    main_node_ip=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
    num_nodes=$(scontrol show hostnames $SLURM_JOB_NODELIST | wc -l)
    num_gpus_per_node=$(python -c "import os; print(len(os.environ['CUDA_VISIBLE_DEVICES'].split(',')))")
    num_gpus=$(($num_nodes * $num_gpus_per_node))
else
    # Running directly
    echo "Running directly" >&2
    # get num of GPUs
    num_gpus=$(nvidia-smi --query-gpu=gpu_name --format=csv,noheader | wc -l)
    echo "Number of GPUs is $num_gpus" >&2
    main_node_ip=$ip_addr
    num_gpus_per_node=$num_gpus
    num_nodes=1
fi

# Debug info (redirected to stderr so it doesn't interfere with output)
echo "Main Node IP is $main_node_ip" >&2
echo "IP Address is $ip_addr" >&2
echo "Master Port is $PORT" >&2
echo "Total Number of nodes is $num_nodes" >&2
echo "GPU per node is $num_gpus_per_node, Total Number of GPUs is $num_gpus" >&2

# if num_nodes is larger than 1, use distributed
if [ $num_nodes -gt 1 ]; then
    nodes=( $( scontrol show hostnames $SLURM_JOB_NODELIST ) )
    nodes_array=($nodes)
    head_node=${nodes_array[0]}
    head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)
    accelerate_cmd="
    srun \
    torchrun \
    --nnodes $SLURM_NNODES \
    --nproc_per_node 1 \
    --rdzv_id $RANDOM \
    --rdzv_backend c10d \
    --rdzv_endpoint $head_node_ip:29500 \
    "
else
    accelerate_cmd="
    accelerate launch --config_file configs/accelerate/default.yaml \
    --main_process_port $PORT \
    --num_processes $num_gpus \
    --num_machines $num_nodes \
    $distributed_type \
    "
fi

# Check if script is being sourced or executed
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    # Script is being executed directly - output only the accelerate_cmd
    echo "$accelerate_cmd"
else
    # Script is being sourced - export variables
    export accelerate_cmd
    export PORT
    export num_gpus
    export num_nodes
    export num_gpus_per_node
    export main_node_ip
    export ip_addr
fi 