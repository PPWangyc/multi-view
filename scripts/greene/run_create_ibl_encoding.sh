#!/bin/bash

while IFS= read -r line
do
    echo "Create dataset on ses eid: $line"
    sbatch create_ibl_encoding.sh 1 $line
    
done < "../../data/encoding_test_eids.txt"