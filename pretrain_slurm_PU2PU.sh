#!/bin/bash
#SBATCH --job-name=PU2PU_DA
#SBATCH --output=PU2PU.txt
#SBATCH --error=PU2PU.log
#SBATCH --mail-type=ALL
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=gpucompute
#SBATCH --mem=10GB
#SBATCH --gres=gpu:1

SRC="$1" # source domain
TAR="$2" # target domain

module load cuda11.1/toolkit/11.1.1
srun python src/main.py --src_data="PU" --tar_data="PU" --src_domain=$SRC --tar_domain=$TAR --lr=0.0001 \
						--batch_size=128 --epochs=100  --input_time_dim=65 --input_freq_dim=18 --input_channel=1 --num_classes=7 \
						--pretrained \
						--use_domain_adv \
						--use_tar_entropy \
						#--warmup_epoch=20

#srun python src/main.py --src_domain=$SRC --tar_domain=$TAR --lr=0.005 --batch_size=128 --epochs=600 --pretrained \
#        --pretrained_path=$MODELPATH
#srun python tests/src/test_cuda.py




