#!/bin/bash

while IFS= read -r line
do
    echo "Create dataset on ses eid: $line"
    sbatch create_ibl_encoding.sh $line
    
done < "../../data/encoding_test_eids.txt"