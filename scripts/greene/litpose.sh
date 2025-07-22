#!/bin/bash

#SBATCH --account=pr_136_tandon_advanced
#SBATCH --job-name="litpose"
#SBATCH --output="litpose.%j.out"
#SBATCH -N 1
#SBATCH -c 8
#SBATCH --ntasks-per-node=1
#SBATCH --mem=64g
#SBATCH --gres=gpu:1
#SBATCH -t 48:00:00
#SBATCH --export=ALL


. ~/.bashrc
cd ../..
conda activate mv


model=$1
train_frame=$2
seed=$3

dataset=mirror-mouse-separate
epochs=300
mode=ft
script_args="
    --model $model \
    --litpose_frame $train_frame \
    --seed $seed \
    --litpose_config configs/litpose/config_${dataset}.yaml \
    --output_dir configs/litpose \
    --epochs ${epochs} \
    --mode ${mode} \
    --dataset ${dataset}
"
generated_config=$(python litpose/edit_config.py $script_args)
litpose_args="
--output_dir outputs/ds-${dataset}_mode-${mode}_model-${model}_frame-${train_frame}_epoch-${epochs}_seed-${seed} \
"
# Now use $generated_config in subsequent commands
echo "Generated config file: $generated_config"
echo $litpose_args
litpose train $generated_config $litpose_args
cd scripts/greene
conda deactivate