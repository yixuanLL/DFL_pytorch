#!/bin/bash
#SBATCH --job-name=stand_cpl
#SBATCH --output=out_cpl_stand
#SBATCH --gres=gpu:1
#SBATCH --mem=16GB
# cur_path=`pwd`



data="SVHN"


dir_path=$(dirname $(pwd))
echo "${dir_path}"
cur_date="`date +%Y%m%d`" 

logfile_path=${dir_path}/logs/
logfile=${dir_path}/logs/standalone/log_cpl_8_${data}_$cur_date
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
lr=(0.1)
batch_size=(256)
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
                    # py_req="python ${dir_path}/main_stand.py --local_round=${round} --global_round=${round} --lr=${l} --dataset=${data} ${k} --opt=${o} --num_clients=1 --momentum=${m} --batch_size=${b}";
                    # py_req="python ${dir_path}/main_stand.py --cpl=T --grad_perp_norm=${gn} --eps=${e} --local_round=${round} --global_round=${round} --lr=${l} --dataset=CIFAR10 --model=cnn5 ${k} --opt=${o}  --num_clients=1 --momentum=${m}  --batch_size=256";
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
done


#CIFAR10
## DP eps=1
# sgd; lr=4 mom=0.9 norm=0.005 50%
round=20
seed=(0)
momentum=(0.0)
lr=(0.1)
g_perp_norm=(0.01 0.1 0.5 1)
eps=(3)
opt=('sgd')

output=0
py_req='0'
echo "====DP====">>$logfile
for m in ${momentum[@]}
do
    for e in ${eps[@]}
    do
        for l in ${lr[@]}
        do
            for gpn in ${g_perp_norm[@]}
            do
                # py_req="python ${dir_path}/main_stand.py --cpl=T --grad_perp_norm=${gpn} --dp=True --eps=${e} --local_round=${round} --global_round=${round} --lr=${l} --dataset=${data} --num_clients=1 --momentum=${m}  --batch_size=${b}";
                # py_req="python ${dir_path}/main_stand.py --cpl=T --grad_perp_norm=${gpn} --dp=True --eps=3.0 --local_round=${round} --global_round=${round} --lr=${l} --dataset=MNIST --num_clients=1 --momentum=${m}  --batch_size=256 --noise_multiplier_g=0.803 --noise_multiplier_p=0.81 --noise_multiplier_a=2.0";
                # py_req="python ${dir_path}/main_stand.py --cpl=T --grad_perp_norm=${gpn} --dp=True --eps=3.0 --local_round=${round} --global_round=${round} --lr=${l} --dataset=SVHN --num_clients=1 --momentum=${m}  --batch_size=128 --noise_multiplier_g=0.695 --noise_multiplier_p=0.696 --noise_multiplier_a=2.0";
                # py_req="python ${dir_path}/main_stand.py --cpl=T --grad_perp_norm=${gpn} --dp=True --eps=3.0 --local_round=${round} --global_round=${round} --lr=${l} --dataset=CIFAR10 --num_clients=1 --momentum=${m}  --batch_size=256 --noise_multiplier_g=0.835 --noise_multiplier_p=0.84 --noise_multiplier_a=3.0";

                # py_req="python ${dir_path}/main_stand.py --cpl=T --grad_perp_norm=${gpn} --dp=True --eps=8.0 --local_round=${round} --global_round=${round} --lr=${l} --dataset=MNIST --num_clients=1 --momentum=${m}  --batch_size=256 --noise_multiplier_g=0.59 --noise_multiplier_p=0.6 --noise_multiplier_a=0.8";
                # py_req="python ${dir_path}/main_stand.py --cpl=T --grad_perp_norm=${gpn} --dp=True --eps=8.0 --local_round=${round} --global_round=${round} --lr=${l} --dataset=CIFAR10 --num_clients=1 --momentum=${m}  --batch_size=256 --noise_multiplier_g=0.605 --noise_multiplier_p=0.61 --noise_multiplier_a=1.0";
                py_req="python ${dir_path}/main_stand.py --cpl=T --grad_perp_norm=${gpn} --dp=True --eps=8.0 --local_round=${round} --global_round=${round} --lr=${l} --dataset=SVHN --num_clients=1 --momentum=${m}  --batch_size=128 --noise_multiplier_g=0.527 --noise_multiplier_p=0.531 --noise_multiplier_a=0.8";

                # py_req="python ${dir_path}/main_stand.py --cpl=T --grad_perp_norm=${gpn} --dp=True --eps=1.0 --local_round=${round} --global_round=${round} --lr=${l} --dataset=MNIST --num_clients=1 --momentum=${m}  --batch_size=256 --noise_multiplier_g=1.4 --noise_multiplier_p=1.47 --noise_multiplier_a=3.5";
                # py_req="python ${dir_path}/main_stand.py --cpl=T --grad_perp_norm=${gpn} --dp=True --eps=1.0 --local_round=${round} --global_round=${round} --lr=${l} --dataset=CIFAR10 --num_clients=1 --momentum=${m}  --batch_size=256 --noise_multiplier_g=1.5 --noise_multiplier_p=1.57 --noise_multiplier_a=4.0";
                # py_req="python ${dir_path}/main_stand.py --cpl=T --grad_perp_norm=${gpn} --dp=True --eps=1.0 --local_round=${round} --global_round=${round} --lr=${l} --dataset=SVHN --num_clients=1 --momentum=${m}  --batch_size=128 --noise_multiplier_g=1.075 --noise_multiplier_p=1.08 --noise_multiplier_a=3.5";

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