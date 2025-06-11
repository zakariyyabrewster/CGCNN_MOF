#!/bin/bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$DIR" # root/scripts
cd .. # root


# Create the target directory if it doesn't exist
mkdir -p test_datasets/CoRE2019

# Extract the tar file to the target directory
tar -xf test_datasets/CoRE2019.tar -C test_datasets/CoRE2019

# TARGETS=("Di" "CH4_HP" "CO2_LP" "logKH_CO2")
TARGETS=$@

# Loop over target properties
for TARGET in "${TARGETS[@]}"; do
    echo "=============================="
    echo "Training on target: $TARGET"
    echo "=============================="

    # Run training script with the current target_property
    python train.py --config config_ft_cgcnn.yaml --target_prop $TARGET

    echo "Finished training on target: $TARGET"
    echo ""
done
# python finetune_cgcnn.py --config config_ft_cgcnn.yaml --target_prop Di 