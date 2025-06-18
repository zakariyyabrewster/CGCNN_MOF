#!/bin/bash

#SBATCH --account=def-moosavi5
#SBATCH --job-name=cgcnn_run
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=0-00:30
#SBATCH --gres=gpu:1

module load python/3.11.5
module load cuda/11.8.0
module load cudnn/8.7.0.84

source $REPO_DIR/myenv/bin/activate

bash extract_db.sh # extract tar file of CoRE2019 
bash train_cgcnn.sh Di # train model on Di
