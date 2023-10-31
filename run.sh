#!/bin/bash
#SBATCH --job-name=slurmtest
#SBATCH --output=out_slurmtest
#SBATCH --gres=gpu:2
#SBATCH --mem=1GB


python /home/yliu270/workspace/DFL_pytorch/main_flamby.py --seed=0 --cpl=True --dp=True --eps=1 --grad_perp_norm=0.2 --local_round=2 --global_round=50 --lr=0.5 --batch_size=2 --dataset=FLamby --model=mclr --clip_paral=0.05

