#!/bin/bash
#SBATCH --job-name=reflownet
#SBATCH --output=reflownet_output.txt
#SBATCH --error=reflownet_error.log
#SBATCH --mail-type=ALL
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=gpucompute
#SBATCH --mem=20GB
#SBATCH --gres=gpu:1

module load cuda11.1/toolkit/11.1.1

WANdB_APIT_KEY=a78d9834d7b1cc2bacf2a7aca59aed42b4b2cd78

RAW_DATA_PATH=$1
PROCESSED_DATA_PATH=$2
TEST_RECIEP=$3
SRC_RATIO=$4

srun -n1 --gpus=1 --exclusive -c1 python src/data/make_dataset.py $1 data/processed/$2 --test_recipe=$3 --src_p=$4 --no_tar_geom=$5
srun -n1 --gpus=1 --exclusive -c1 python src/pretrain_model.py data/processed/$2 --log=$2 --epoch_size=100
srun -n1 --gpus=1 --exclusive -c1 python src/train_model.py data/processed/$2 --log=$2 --epoch_size=100
srun -n1 --gpus=1 --exclusive -c1 python src/predict_model.py data/processed/$2 --log=$2
