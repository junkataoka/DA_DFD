#!/bin/bash
#SBATCH --job-name=DA_DFD_train
#SBATCH --output=DA_DFD.txt
#SBATCH --error=DA_DFD.log
#SBATCH --mail-type=ALL
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=gpucompute
#SBATCH --mem=10GB
#SBATCH --gres=gpu:1

SRC="$1" # source domain
TAR="$2" # target domain

module load cuda11.1/toolkit/11.1.1
srun python src/main.py --src_data="CWRU" --tar_data="CWRU" --src_domain=$SRC --tar_domain=$TAR --lr=0.00001 \
                        --batch_size=128 --epochs=200 --pretrained --use_domain_adv \
                        #--use_contra_learn --accum_iter=5




