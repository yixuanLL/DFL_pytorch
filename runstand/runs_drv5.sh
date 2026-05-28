#!/bin/bash

data="MNIST"

dir_path=$(dirname $(pwd))
echo "${dir_path}"
cur_date="`date +%Y%m%d`" 

logfile_path=${dir_path}/logs/
logfile=${dir_path}/logs/standalone/log_drv5_${data}_$cur_date
if [ ! -x $logfile_path ]; then
 mkdir "$logfile_path"
fi
# source /local/scratch/yliu270/anaconda3/bin/activate flamby
source /Users/liu/miniconda3/bin/activate venv
if [ ! -f "$logfile" ]; then
 touch "$logfile"
fi

time=$(date "+%Y-%m-%d %H:%M:%S")
echo "${time}">>$logfile

echo "====DR=True; V5 ====">>$logfile

round=20
momentum=0.0
seed=(0)
lr=(2)
g_p_norm=(0.2)
clip_paral=(0.05) # alpha actucally
index=(0)
eps=(2.98)
eps_2=(0.02)
iidflag=("--save_dir=result")
opt=('sgd')
batch=(256)

output=0
echo "====DP: all the steps use DPDR====">>$logfile
for b in ${batch[@]}
do
    for s in ${seed[@]}
    do
        for cp in ${clip_paral[@]}
        do
            for l in ${lr[@]}
            do
                for gpn in ${g_p_norm[@]}
                do
                    py_req="python ${dir_path}/main_stand.py --seed=${s} --DR=True --grad_norm=5 --grad_perp_norm=${gpn} --dp=True --eps=3  --local_round=${round} --global_round=${round} --lr=${l}  --dataset=MNIST --clip_paral=${cp} --num_clients=1 --batch_size=256  --noise_multiplier_g=0.803 --noise_multiplier_p=0.81 --noise_multiplier_a=2.0";
                    # py_req="python ${dir_path}/main_stand.py --seed=${s} --DR=True --grad_norm=5 --grad_perp_norm=${gpn} --dp=True --eps=3  --local_round=${round} --global_round=${round} --lr=${l}  --dataset=CIFAR10 --clip_paral=${cp} --num_clients=1 --batch_size=256  --noise_multiplier_g=0.835 --noise_multiplier_p=0.84 --noise_multiplier_a=3.0";
                    # py_req="python ${dir_path}/main_stand.py --seed=${s} --DR=True --grad_norm=5 --grad_perp_norm=${gpn} --dp=True --eps=3  --local_round=${round} --global_round=${round} --lr=${l}  --dataset=SVHN --clip_paral=${cp} --num_clients=1 --batch_size=128 --noise_multiplier_g=0.695 --noise_multiplier_p=0.696 --noise_multiplier_a=2.0";

                    # py_req="python ${dir_path}/main_stand.py --seed=${s} --DR=True --grad_norm=5 --grad_perp_norm=${gpn} --dp=True --eps=8  --local_round=${round} --global_round=${round} --lr=${l}  --dataset=MNIST --clip_paral=${cp} --num_clients=1 --batch_size=256  --noise_multiplier_g=0.59 --noise_multiplier_p=0.6 --noise_multiplier_a=0.8";
                    # py_req="python ${dir_path}/main_stand.py --seed=${s} --DR=True --grad_norm=5 --grad_perp_norm=${gpn} --dp=True --eps=8  --local_round=${round} --global_round=${round} --lr=${l}  --dataset=CIFAR10 --clip_paral=${cp} --num_clients=1 --batch_size=256  --noise_multiplier_g=0.605 --noise_multiplier_p=0.61 --noise_multiplier_a=1.0";
                    # py_req="python ${dir_path}/main_stand.py --seed=${s} --DR=True --grad_norm=5 --grad_perp_norm=${gpn} --dp=True --eps=8  --local_round=${round} --global_round=${round} --lr=${l}  --dataset=SVHN --clip_paral=${cp} --num_clients=1 --batch_size=128 --noise_multiplier_g=0.527 --noise_multiplier_p=0.531 --noise_multiplier_a=0.8";

                    # py_req="python ${dir_path}/main_stand.py --seed=${s} --DR=True --grad_norm=5 --grad_perp_norm=${gpn} --dp=True --eps=1  --local_round=${round} --global_round=${round} --lr=${l}  --dataset=MNIST --clip_paral=${cp} --num_clients=1 --batch_size=256  --noise_multiplier_g=1.4 --noise_multiplier_p=1.47 --noise_multiplier_a=3.5";
                    # py_req="python ${dir_path}/main_stand.py --seed=${s} --DR=True --grad_norm=5 --grad_perp_norm=${gpn} --dp=True --eps=1  --local_round=${round} --global_round=${round} --lr=${l}  --dataset=CIFAR10 --clip_paral=${cp} --num_clients=1 --batch_size=256  --noise_multiplier_g=1.5 --noise_multiplier_p=1.57 --noise_multiplier_a=4.0";
                    # py_req="python ${dir_path}/main_stand.py --seed=${s} --DR=True --grad_norm=5 --grad_perp_norm=${gpn} --dp=True --eps=1  --local_round=${round} --global_round=${round} --lr=${l}  --dataset=SVHN --clip_paral=${cp} --num_clients=1 --batch_size=128 --noise_multiplier_g=1.075 --noise_multiplier_p=1.08 --noise_multiplier_a=3.5";

                    # py_req="python ${dir_path}/main_stand.py --DR=True --eps=${e} --grad_norm=2 --grad_perp_norm=${gpn} --dp=True --local_round=${round} --global_round=${round} --lr=${l} --dataset=${data} --model=mclr  --clip_paral=${cp} --opt=${o} --num_clients=1 --momentum=${momentum} --batch_size=32";
                    # py_req="python ${dir_path}/main_test.py --DRtest=True --eps=${e} --grad_perp_norm=${gpn} --dp=True --local_round=2 --global_round=100 --lr=${l} --dataset=MNIST --model=cnn";
                    # py_req="python ${dir_path}/main_flamby.py --seed=${s} --DRtest=True --dp=True --eps=${e} --grad_perp_norm=${gpn} --local_round=2 --global_round=50 --lr=${l} --batch_size=2 --eps_2=0.02 --dataset=FLamby --model=mclr --clip_p=0.001 ${k}";
                    echo "${py_req}"
                    echo "${py_req}">>$logfile
                    start_time=$(date +%s)
                    output=`${py_req}`;
                    end_time=$(date +%s)
                    if [ $? -ne 0 ]; then
                        echo "[FAILED] ${py_req}"
                        echo "[FAILED] ${py_req}">>$logfile
                        exit 8
                    fi
                    sleep 1;
                    echo "${output}">>$logfile
                    cost_time=$[ $end_time-$start_time ]
                    time=$(date "+%H:%M:%S")
                    echo "${time}">>$logfile
                    echo "[time] build py time is $(($cost_time/60))min $(($cost_time%60))s"
                    echo "[time] build py time is $(($cost_time/60))min $(($cost_time%60))s">>$logfile
                done
            done
        done
    done
done


time=$(date "+%Y-%m-%d %H:%M:%S")
echo "${time}">>$logfile
echo "[finished]!"
echo "[finished]!">>$logfile