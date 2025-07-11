#!/bin/bash

#SBATCH --account=def-moosavi5
#SBATCH --job-name=kcv_cgcnn_run
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --time=0-10:00
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

# bash extract_db.sh

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

