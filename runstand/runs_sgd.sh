#!/bin/bash
#SBATCH --job-name=stand_sgd_cifar
#SBATCH --output=out_sgd_cifar_stand
#SBATCH --gres=gpu:1
#SBATCH --mem=8GB
# cur_path=`pwd`
dir_path=$(dirname $(pwd))
echo "${dir_path}"
cur_date="`date +%Y%m%d`" 

logfile_path=${dir_path}/logs/
logfile=${dir_path}/logs/standalone/log_sgd_cifar_$cur_date
if [ ! -x $logfile_path ]; then
 mkdir "$logfile_path"
fi

if [ ! -f "$logfile" ]; then
 touch "$logfile"
fi
source /home/yliu270/anaconda3/bin/activate flamby

#MNIST
# seed=(0)
# momentum=(0.0)
# lr=(0.1 0.2)
# g_norm=(0.1 0.2 0.5 1.0)
# eps=(0.5 3)
# iidflag=("--save_dir=result")
# kf=("--kf=True")
# seed=(0)
# lr=(0.1 0.2 0.5)
# g_norm=(0.01 0.1 0.5)
# eps=(0.1 0.5 1)
# iidflag=("--save_dir=result")
# kf=("--save_dir=result")
# momentum=(0.0)


#CIFAR10
## non-DP
# adam: lr=0.001 61
# mom: lr=0.01 mom=0.5 64
# sgd: lr=0.01 20rounds 72
## DP eps=1
# sgd; lr=4 mom=0.9 norm=0.005 50%
round=20
seed=(0)
momentum=(0.0 0.5 0.9)
lr=(0.01 0.1 1)
g_norm=(100)
kf=("--save_dir=result")
iidflag=("--save_dir=result")
opt=('sgd')

time=$(date "+%Y-%m-%d %H:%M:%S")
echo "${time}">>$logfile

echo "====NoDP====">>$logfile
for o in ${opt[@]}
do
    for m in ${momentum[@]}
    do
        for l in ${lr[@]}
        do
            for gn in ${g_norm[@]}
            do
                py_req="python ${dir_path}/main_stand.py --grad_norm=${gn} --local_round=${round} --global_round=${round} --lr=${l} --dataset=CIFAR10 --model=cnn5 ${k} --opt=${o} --num_clients=1 --momentum=${m} --batch_size=256";
                # py_req="python ${dir_path}/main_stand.py --grad_norm=${gn} --local_round=${round} --global_round=${round} --lr=${l} --dataset=MNIST --model=cnn ${k}  --num_clients=1";
                echo "${py_req}"
                echo "FedSVG without DP, but with clip">>$logfile
                echo "${py_req}">>$logfile
                start_time=$(date +%s)
                # output=`${py_req}`;
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


#CIFAR10
## DP eps=1
# sgd; lr=4 mom=0.9 norm=0.005 50%
round=20
seed=(0)
momentum=(0.0)
lr=(0.001)
g_norm=(1 1.5 2)
eps=(1 3)
kf=("--save_dir=result")
iidflag=("--save_dir=result")
opt=('adam')

output=0
echo "====DP====">>$logfile
for o in ${opt[@]}
do
    for m in ${momentum[@]}
    do
        for e in ${eps[@]}
        do
            for l in ${lr[@]}
            do
                for gn in ${g_norm[@]}
                do
                    py_req="python ${dir_path}/main_stand.py --grad_norm=${gn} --dp=True --eps=${e} --local_round=${round} --global_round=${round} --lr=${l} --dataset=CIFAR10 --model=cnn5 ${k} --opt=${o}  --num_clients=1 --momentum=${m}  --batch_size=256";
                    # py_req="python ${dir_path}/main_stand.py --cpl=T --grad_perp_norm=${gn} --dp=True --eps=${e} --local_round=${round} --global_round=${round} --lr=${l} --dataset=CIFAR10 --model=cnn5 ${k} --opt=${o}  --num_clients=1 --momentum=${m}  --batch_size=256";
                    # py_req="python ${dir_path}/main_stand.py --grad_norm=${gn} --dp=True --eps=${e}  --local_round=2 --global_round=100 --lr=${l} --dataset=MNIST --model=cnn  ${k}";
                    # py_req="python ${dir_path}/main_stand.py --seed=${s} --dp=True --eps=${e} --grad_norm=${gn} --local_round=2 --global_round=50 --lr=${l} --batch_size=2  --dataset=FLamby --model=mclr ${k}";
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