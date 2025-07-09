#!/bin/bash

# TARGETS=("Di" "CH4_HP" "CO2_LP" "logKH_CO2")
TARGETS=("$@") # Accept target properties as command line arguments

# Loop over target properties
for TARGET in "${TARGETS[@]}"; do
    echo "=============================="
    echo "kCV on target: $TARGET"
    echo "=============================="

    # Run training script with the current target_property
    python kcv_cgcnn.py --target_property $TARGET
    echo "Finished kCV on target: $TARGET"
done