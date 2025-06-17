#!/bin/bash

#SBATCH --account=def-moosavi5
#SBATCH --job-name=cgcnn_run
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=0-00:30
#SBATCH --gres=gpu:1

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$DIR" # root/scripts
cd .. # root

bash scripts/extract_db.sh # extract tar file of CoRE2019 
bash scripts/train_cgcnn.sh Di # train model on Di
