#!/bin/bash
#SBATCH --job-name=DA_DFD_pretrain
#SBATCH --output=DA_DFD.txt
#SBATCH --error=DA_DFD.log
#SBATCH --mail-type=ALL
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=gpucompute
#SBATCH --mem=10GB
#SBATCH --gres=gpu:1
####SBATCH --nodelist=compute130

SRC="$1" # source domain
TAR="$2" # target domain

module load cuda11.1/toolkit/11.1.1

srun python src/main.py --src_data="CWRU_small" --tar_data="CWRU_small" --src_domain=$SRC --tar_domain=$TAR --lr=0.0001 \
                        --batch_size=128 --epochs=200 --input_channel=1 --num_classes=4 \
                        --model="wdcnn" \
                        #--pretrained \
                        #--use_domain_adv \
                        #--use_tar_entropy \
                        #--warmup_epoch=20 \
                        #--use_domain_bn \

#srun python src/main.py --src_domain=$SRC --tar_domain=$TAR --lr=0.005 --batch_size=128 --epochs=600 --pretrained \
#        --pretrained_path=$MODELPATH
#srun python tests/src/test_cuda.py




