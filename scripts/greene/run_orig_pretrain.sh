#!/bin/bash

memory=64g
num_nodes=2
constraint="a100|h100|rtx8000"
multi_node_args="
--cpus-per-task=16
--gres=gpu:1
--constraint=$constraint
--ntasks=$num_nodes
--nodes=$num_nodes
--mem=$memory
--time=48:00:00
--output=pretrain.%j.out
"

script_path=pretrain.sh

sbatch $multi_node_args $script_path