#!/bin/bash
#SBATCH --job-name=autoclip
#SBATCH --output=out_autoclip
#SBATCH --gres=gpu:1
#SBATCH --mem=16GB
# cur_path=`pwd`



data="CIFAR10"


dir_path=$(dirname $(pwd))
echo "${dir_path}"
cur_date="`date +%Y%m%d`" 

logfile_path=${dir_path}/logs/
logfile=${dir_path}/logs/standalone/log_autoclip_${data}_$cur_date
if [ ! -x $logfile_path ]; then
 mkdir "$logfile_path"
fi

if [ ! -f "$logfile" ]; then
 touch "$logfile"
fi
# source /home/yliu270/anaconda3/bin/activate flamby
source /local/scratch/yliu270/anaconda3/bin/activate flamby

time=$(date "+%Y-%m-%d %H:%M:%S")
echo "${time}">>$logfile


global_round=(20)
seed=(0)
momentum=(0.0)
lr=(0.1 0.2 0.5)
batch_size=(128)
opt=('sgd')


py_req='0'
echo "====NoDP====">>$logfile
for round in ${global_round[@]}
do
    for o in ${opt[@]}
    do
        for m in ${momentum[@]}
        do
            for l in ${lr[@]}
            do
                for b in ${batch_size[@]}
                do
                    # py_req="python ${dir_path}/main_stand.py --local_round=${round} --global_round=${round} --lr=${l} --dataset=${data} --opt=${o} --num_clients=1 --momentum=${m} --batch_size=${b}";
                    # py_req="python ${dir_path}/main_stand.py --cpl=T --grad_perp_norm=${gn} --eps=${e} --local_round=${round} --global_round=${round} --lr=${l} --dataset=CIFAR10 --model=cnn5 ${k} --opt=${o}  --num_clients=1 --momentum=${m}  --batch_size=256";
                    echo "${py_req}"
                    echo "FedSVG without DP, but with clip">>$logfile
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
                    echo "[time] build py time is $(($cost_time/60))min $(($cost_time%60))s"
                    echo "[time] build py time is $(($cost_time/60))min $(($cost_time%60))s">>$logfile
                done
            done
        done
    done
done


#CIFAR10
## DP eps=1
# sgd; lr=4 mom=0.9 norm=0.005 50%
round=20
seed=(0)
momentum=(0.0)
lr=(2)
grad_norm=(0.8 1 1.5 2)
# eps=(3)
# opt=('sgd')
index=(2)
batch=(64 256 1024 4096)
noise_multiplier_g=(0.666 0.84 1.23 2.14)
noise_multiplier_p=(0.667 0.84 1.24 2.2)
noise_multiplier_a=(2 3 6 8)

output=0
py_req='0'
echo "====DP====">>$logfile
for id in ${index[@]}
do
    for s in ${seed[@]}
    do
        for l in ${lr[@]}
        do
            for gn in ${grad_norm[@]}
            do
                py_req="python ${dir_path}/main_stand.py --seed=${s} --grad_norm=${gn} --dp=True --alg=autoclip --eps=3 --local_round=${round} --global_round=${round} --lr=${l} --dataset=CIFAR10  --opt=sgd --num_clients=1  --batch_size=${batch[${id}]}  --noise_multiplier_g=${noise_multiplier_g[${id}]} --noise_multiplier_p=${noise_multiplier_p[${id}]} --noise_multiplier_a=${noise_multiplier_a[${id}]}";

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


time=$(date "+%Y-%m-%d %H:%M:%S")
echo "${time}">>$logfile
echo "[finished]!"
echo "[finished]!">>$logfile