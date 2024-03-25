#!/bin/bash
#SBATCH --job-name=test
#SBATCH --output=out_test
#SBATCH --gres=gpu:1
#SBATCH --mem=24GB

source /local/scratch/yliu270/anaconda3/bin/activate flamby
start_time=$(date +%s)

py_req="python /local/scratch/yliu270/workspace/DFL_pytorch/main_stand.py 
--DRtest=True --eps=100 --eps_2=100 --grad_norm=5 --grad_perp_norm=0.3 
--dp=True --local_round=2 --global_round=2 --lr=0.1 --dataset=SVHN  
--clip_paral=0.1 --opt=sgd --num_clients=1 --momentum=0.0 --batch_size=128"

# py_req="python /local/scratch/yliu270/workspace/DFL_pytorch/main_stand_plot.py"

echo "${py_req}"
output=`${py_req}`;
echo "${output}"
end_time=$(date +%s)
cost_time=$[ $end_time-$start_time ]
echo "[time] build py time is $(($cost_time/60))min $(($cost_time%60))s"
