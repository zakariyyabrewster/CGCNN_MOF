#!/bin/bash

# 5-Fold Cross-Validation script for PointNet on MOF property prediction
# Similar to kcv_cgcnn_run.sh and kcv_transformer_run.sh

echo "Starting 5-Fold Cross-Validation for PointNet on MOF properties..."

# List of properties to evaluate
properties=("Di" "Df" "Dif" "CH4_HP" "CO2_LP" "logKH_CO2")

# Random seed

for prop in "${properties[@]}"; do
    echo "=========================================="
    echo "Running 5-Fold CV for property: $prop"
    echo "=========================================="
    
    python kcv_pcd.py --target_property $prop
done 
echo "All 5-Fold Cross-Validation experiments completed!"
echo "Results saved in training_results/finetuning/PointNet_CV/"
