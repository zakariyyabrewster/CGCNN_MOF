#!/bin/bash

#SBATCH --account=def-moosavi5
#SBATCH --job-name=llm_run
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
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

echo "Running finetune_llm.py"
python finetune_llm.py
echo "Finished running finetune_llm.py"

