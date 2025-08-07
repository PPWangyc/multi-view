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

dataset=$1 # fly-anipose, mirror-mouse-separate
model=$2 # resnet50_animal_ap10k, vitb_imagenet, vitb-mv, vitb-sv
train_frame=$3 # 100, 1000
seed=$4 # 0, 1, 2

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