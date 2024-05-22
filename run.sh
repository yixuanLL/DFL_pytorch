#!/bin/bash
#SBATCH --job-name=test
#SBATCH --output=out_test
#SBATCH --gres=gpu:1
#SBATCH --mem=24GB

source /local/scratch/yliu270/anaconda3/bin/activate flamby
start_time=$(date +%s)

py_req="python /local/scratch/yliu270/workspace/DFL_pytorch/main_stand.py --seed=0 --grad_norm=0.15 --dp=True --eps=3 --local_round=20 --global_round=20 --lr=2 --dataset=SVHN  --opt=sgd --num_clients=1 --batch_size=128  --noise_multiplier_g=0.695 --noise_multiplier_p=0.696 --noise_multiplier_a=2.0"

# py_req="python /local/scratch/yliu270/workspace/DFL_pytorch/main_stand_save.py"



echo "${py_req}"
output=`${py_req}`;
echo "${output}"
end_time=$(date +%s)
cost_time=$[ $end_time-$start_time ]
echo "[time] build py time is $(($cost_time/60))min $(($cost_time%60))s"
