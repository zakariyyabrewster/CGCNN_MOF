#!/bin/bash

#SBATCH --account=def-moosavi5
#SBATCH --job-name=cgcnn_run
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=0-00:30
#SBATCH --gres=gpu:1

REPO_DIR=/home/brewst27/projects/def-moosavi5/brewst27/CGCNN_MOF
cd $REPO_DIR

module load python/3.13.2
module load cuda/12.2
module load cudnn/

source $REPO_DIR/myenv/bin/activate

bash scripts/extract_db.sh # extract tar file of CoRE2019 
bash scripts/train_cgcnn.sh Di # train model on Di
