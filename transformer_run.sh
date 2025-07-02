#!/bin/bash

#SBATCH --account=def-moosavi5
#SBATCH --job-name=transformer_run
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=0-02:00
#SBATCH --gres=gpu:1

cd $SLURM_SUBMIT_DIR

module load StdEnv/2023
module load gcc/12.3
module load r-bundle-bioconductor/3.20
module load python/3.11.5
module load cudacore/.12.6.2
module load cuda/12.6.0
module load cudnn/9.5.1.17
module load yaml/2.3.10


source myenv/bin/activate

bash train_transformer.sh Di # train model on Di, CH4_HP, CO2_LP, logKH_CO2
