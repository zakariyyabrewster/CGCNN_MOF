#!/bin/bash

#SBATCH --account=def-moosavi5
#SBATCH --job-name=cgcnn_run
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=0-00:30
#SBATCH --gres=gpu:1

cd $SLURM_SUBMIT_DIR

module load StdEnv/2023
module load python/3.11.5
module load cudacore/.12.6.2
module load cuda/12.6.0
module load cudnn/9.5.1.17
module load yaml/2.3.10


source myenv/bin/activate

bash extract_db.sh # extract tar file of CoRE2019 
bash train_cgcnn.sh Di # train model on Di
