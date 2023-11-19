#!/bin/bash
#SBATCH --job-name=slurmtest
#SBATCH --output=out_slurmtest
#SBATCH --gres=gpu:2
#SBATCH --mem=1GB

source /home/yliu270/anaconda3/bin/activate flamby
python /home/yliu270/workspace/DFL_pytorch/main_test.py --grad_norm=1.0 --dp=True --eps=3  --local_round=2 --global_round=100 --lr=0.2 --dataset=MNIST --model=cnn  --save_dir=result


