#!/bin/bash

model=$1
while IFS= read -r line
do
    echo "IBL-Encoding on ses eid: $line with model: $model"
    sbatch ibl_encoding.sh $line $model

done < "../../data/encoding_test_eids.txt"