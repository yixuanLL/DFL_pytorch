#!/bin/bash
#SBATCH --job-name=test
#SBATCH --output=out_test
#SBATCH --gres=gpu:1
#SBATCH --mem=8GB

source /home/yliu270/anaconda3/bin/activate flamby
start_time=$(date +%s)

# py_req="python /home/yliu270/workspace/DFL_pytorch/main_test.py \
# --eps=1.0 \
# --eps_2=0.02 \
# --grad_perp_norm=0.5 \
# --clip_paral=0.05 \
# --lr=0.01 \
# --momentum=0.9 \
# --DRtest=True \
# --dp=True "

# py_req="python /home/yliu270/workspace/DFL_pytorch/main_test.py \
# --eps=0.3 \
# --eps_2=0.02 \
# --grad_norm=3 \
# --grad_perp_norm=0.2 \
# --clip_paral=0.05 \
# --lr=0.2 \
# --DRtest=True \
# --dp=True"    

# py_req="python /home/yliu270/workspace/DFL_pytorch/main_test.py --DR=True --grad_perp_norm=11 --local_round=2 --global_round=100 --lr=0.2 --dataset=CIFAR10 --model=lenet5 --clip_paral=100"
# py_req="python /home/yliu270/workspace/DFL_pytorch/main_test.py --cpl=True --grad_perp_norm=11 --local_round=2 --global_round=100 --lr=0.2 --dataset=CIFAR10 --model=lenet5"
# python /home/yliu270/workspace/DFL_pytorch/main_test.py --DRtest=True --grad_perp_norm=11 --local_round=2 --global_round=100 --lr=0.2 --dataset=CIFAR10 --model=lenet5 --clip_paral=100

# python /home/yliu270/workspace/DFL_pytorch/main_test.py --grad_norm=0.01 --local_round=2 --global_round=200 --lr=0.2 --dataset=CIFAR10 --model=lenet5 --clip_paral=100 --momentum=0.9
py_req="python /home/yliu270/workspace/DFL_pytorch/main_stand.py --cpl=T --grad_perp_norm=0.01 --dp=True --eps=1 --local_round=20 --global_round=20 --lr=0.1 --dataset=CIFAR10 --model=cnn5  --opt=sgd  --num_clients=1 --momentum=0.0  --batch_size=256
"


output=`${py_req}`;
echo "${output}"
end_time=$(date +%s)
cost_time=$[ $end_time-$start_time ]
echo "[time] build py time is $(($cost_time/60))min $(($cost_time%60))s"
