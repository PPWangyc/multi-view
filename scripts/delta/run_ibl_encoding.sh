#!/bin/bash

model=$1
avail_views=$2 # left+right
while IFS= read -r line
do
    echo "IBL-Encoding on ses eid: $line with model: $model and avail_views: $avail_views"
    sbatch ibl_encoding.sh $line $model $avail_views

done < "../../data/encoding_test_eids.txt"