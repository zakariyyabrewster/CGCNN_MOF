#!/bin/bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$DIR" # root/scripts
cd .. # root


# Create the target directory if it doesn't exist
mkdir -p test_datasets/CoRE2019

# Extract the tar file to the target directory
tar -xf test_datasets/CoRE2019.tar -C test_datasets/CoRE2019

python finetune_cgcnn.py --config config_ft_cgcnn.yaml