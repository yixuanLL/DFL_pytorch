#!/bin/bash
#SBATCH --job-name=test2
#SBATCH --output=out_test2
#SBATCH --gres=gpu:1
#SBATCH --mem=32GB

source /local/scratch/yliu270/anaconda3/bin/activate flamby
start_time=$(date +%s)

py_req="python /local/scratch/yliu270/workspace/DFL_pytorch/main_diff2_stand.py --DRV2=True --eps=3 --seed=0 --grad_norm=5 --grad_perp_norm=0.04 --dp=True --eps=3 --local_round=2 --global_round=2 --lr=2 --dataset=MNIST  --opt=sgd --num_clients=1 --momentum=0.0 --batch_size=128 --noise_multiplier_g=0.695 --noise_multiplier_p=0.696 --noise_multiplier_a=2.0"

echo "${py_req}"
output=`${py_req}`;
echo "${output}"
end_time=$(date +%s)
cost_time=$[ $end_time-$start_time ]
echo "[time] build py time is $(($cost_time/60))min $(($cost_time%60))s"
