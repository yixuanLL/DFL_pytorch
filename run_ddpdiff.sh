#!/bin/bash
#SBATCH --job-name=feddiff
#SBATCH --output=out_feddiff
#SBATCH --gres=gpu:1
#SBATCH --mem=8GB
cur_path=`pwd`

cur_date="`date +%Y%m%d`" 
dataset="CIFAR10"

logfile_path=${cur_path}/logs/
logfile=${cur_path}/logs/log_ddp_diff_${dataset}_$cur_date
if [ ! -x $logfile_path ]; then
 mkdir "$logfile_path"
fi

if [ ! -f "$logfile" ]; then
 touch "$logfile"
fi
source /home/yliu270/anaconda3/bin/activate flamby

#CIFAR10
seed=(0)
sample_ratio=(0.1)
momentum=(0.0)
lr=(0.1 1)
alpha=(1)
global_lr=(0.1 0.3 0.5 1)
gp_norm=(0.05 0.1 0.3 0.5)
index=(0)
eps=(3)
# kf=("--kf=True")
kf=("--save_dir=result")
iidflag=("--save_dir=result")

time=$(date "+%Y-%m-%d %H:%M:%S")
echo "${time}">>$logfile

output=0
echo "====DP====">>$logfile
for a in ${alpha[@]}
do
    for r in ${sample_ratio[@]}
    do
        for glr in ${global_lr[@]}
        do
            for i in ${index[@]}
            do
                for l in ${lr[@]}
                do
                    for gp in ${gp_norm[@]}
                    do
                        # py_req="python ${cur_path}/main_stand.py --grad_norm=${gn} --dp=True --eps=${e}  --local_round=2 --global_round=200 --lr=${l} --dataset=CIFAR10 --model=lenet5 ${k} --momentum=${m}";
                        # py_req="python ${cur_path}/main_fed.py --grad_perp_norm=${gp} --eps=${eps[${i}]} --local_round=1 --global_round=100 --lr=${l}  --glr=${glr} --dataset=${dataset} --model=cnn  --sample_ratio=${r} --num_clients=1000  --FLalg=FedDPDIFF";
                        py_req="python ${cur_path}/main_fed.py --grad_perp_norm=${gp} --eps=${eps[${i}]} --local_round=1 --global_round=100 --lr=${l}  --glr=${glr} --dataset=${dataset} --model=cnn  --sample_ratio=${r} --clip_paral=${a} --num_clients=1000  --FLalg=FedDPDIFF3";
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
done

time=$(date "+%Y-%m-%d %H:%M:%S")
echo "${time}">>$logfile
echo "[finished]!"
echo "[finished]!">>$logfile