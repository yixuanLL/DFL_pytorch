#!/bin/bash
#SBATCH --job-name=slurmtest
#SBATCH --output=out_slurmtest
#SBATCH --gres=gpu:2
#SBATCH --mem=1GB


python /home/yliu270/workspace/DFL_pytorch/main_flamby.py --seed=0 --dp=True --eps=0.3 --grad_norm=0.05 --local_round=2 --global_round=50 --lr=0.1 --batch_size=2  --dataset=FLamby --model=mclr --kf=True

