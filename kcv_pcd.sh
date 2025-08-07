#!/bin/bash

#SBATCH --account=def-moosavi5
#SBATCH --job-name=kcv_pcd
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=20G
#SBATCH --time=0-12:00
#SBATCH --gres=gpu:1

cd $SLURM_SUBMIT_DIR

module load StdEnv/2023
module load gcc/12.3
module load r-bundle-bioconductor/3.20
module load python/3.11.5
module load cudacore/.12.6.2
module load cuda/12.6
module load cudnn/9.5.1.17


source myenv/bin/activate

pip install -r requirements.txt

echo "Starting 5-Fold Cross-Validation for PointNet on MOF properties..."

# List of properties to evaluate
properties=("Di" "Df" "Dif" "CH4_HP" "CO2_LP" "logKH_CO2")

for prop in "${properties[@]}"; do
    echo "=========================================="
    echo "Running 5-Fold CV for property: $prop"
    echo "=========================================="
    
    python kcv_pcd.py --target_property $prop
done 
echo "All 5-Fold Cross-Validation experiments completed!"
echo "Results saved in training_results/finetuning/PointNet_CV/"
