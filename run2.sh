#!/bin/bash
#SBATCH --job-name=test2
#SBATCH --output=out_test2
#SBATCH --gres=gpu:1
#SBATCH --mem=32GB

source /local/scratch/yliu270/anaconda3/bin/activate flamby
start_time=$(date +%s)

py_req="python /local/scratch/yliu270/workspace/DFL_pytorch/main_stand.py --DRtest=True --eps=2.98 --eps_2=0.02 --grad_norm=5 --grad_perp_norm=0.1 --dp=True --local_round=2 --global_round=2 --lr=1 --dataset=SVHN  --clip_paral=0.05 --num_clients=1 --momentum=0.0 --batch_size=128"

echo "${py_req}"
output=`${py_req}`;
echo "${output}"
end_time=$(date +%s)
cost_time=$[ $end_time-$start_time ]
echo "[time] build py time is $(($cost_time/60))min $(($cost_time%60))s"
