#!/bin/bash
#SBATCH --job-name=ddp
#SBATCH --output=out_ddp
#SBATCH --gres=gpu:1
#SBATCH --mem=8GB
cur_path=`pwd`

cur_date="`date +%Y%m%d`" 
dataset="MNIST"

logfile_path=${cur_path}/logs/
logfile=${cur_path}/logs/log_sgd_ddp_${dataset}_$cur_date
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
lr=(0.1)
g_norm=(0.1)
index=(0)
eps=(3)
# kf=("--kf=True")
kf=("--save_dir=result")
iidflag=("--save_dir=result")

time=$(date "+%Y-%m-%d %H:%M:%S")
echo "${time}">>$logfile

echo "====NoDP====">>$logfile
for m in ${momentum[@]}
do
    for r in ${sample_ratio[@]}
    do
        for l in ${lr[@]}
        do
            for k in ${kf[@]}
            do
                # py_req="python ${cur_path}/main_lenet5.py --grad_norm=${gn} --local_round=2 --global_round=200 --lr=${l} --dataset=CIFAR10 --model=lenet5 ${k} --momentum=${m}";
                py_req="python ${cur_path}/main_fed.py --local_round=1 --global_round=500 --lr=${l} --dataset=${dataset} --model=cnn --momentum=${m} --sample_ratio=${r} --num_clients=1000 --FLalg=FedAvg";
                # py_req="python ${cur_path}/main_flamby.py --seed=${s} --grad_norm=${gn} --local_round=2 --global_round=50 --lr=${l} --batch_size=2 --dataset=FLamby --model=mclr ${k}";                
                echo "${py_req}"
                echo "FedAVG without DP, no clip">>$logfile
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

output=0
echo "====DP====">>$logfile
for m in ${momentum[@]}
do
    for r in ${sample_ratio[@]}
    do
        for i in ${index[@]}
        do
            for l in ${lr[@]}
            do
                for g in ${g_norm[@]}
                do
                    # py_req="python ${cur_path}/main_stand.py --grad_norm=${gn} --dp=True --eps=${e}  --local_round=2 --global_round=200 --lr=${l} --dataset=CIFAR10 --model=lenet5 ${k} --momentum=${m}";
                    py_req="python ${cur_path}/main_fed.py --grad_norm=${g} --eps=${eps[${i}]} --local_round=1 --global_round=500 --lr=${l} --dataset=${dataset} --model=cnn --momentum=${m}  --sample_ratio=${r} --num_clients=1000  --FLalg=FedDPAvg";
                    # py_req="python ${cur_path}/main_flamby.py --seed=${s} --dp=True --eps=${e} --grad_norm=${gn} --local_round=2 --global_round=50 --lr=${l} --batch_size=2  --dataset=FLamby --model=mclr ${k}";
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