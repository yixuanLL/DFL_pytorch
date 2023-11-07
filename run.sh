#!/bin/bash
#SBATCH --job-name=slurmtest
#SBATCH --output=out_slurmtest
#SBATCH --gres=gpu:2
#SBATCH --mem=1GB


python /home/yliu270/workspace/DFL_pytorch/main_test.py --dp=True --grad_norm=1 --local_round=5 --global_round=500 --lr=0.05 --dataset=MNIST --model=cnn --num_clients=100 --sample_ratio=0.2 --eps=1 --batch_size=32 --kf=True

