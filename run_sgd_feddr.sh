#!/bin/bash
#SBATCH --job-name=sgdmom
#SBATCH --output=out_sgdmom
#SBATCH --gres=gpu:1
#SBATCH --mem=8GB
cur_path=`pwd`

cur_date="`date +%Y%m%d`" 

logfile_path=${cur_path}/logs/
logfile=${cur_path}/logs/log_sgdmom_$cur_date
if [ ! -x $logfile_path ]; then
 mkdir "$logfile_path"
fi

if [ ! -f "$logfile" ]; then
 touch "$logfile"
fi
source /home/yliu270/anaconda3/bin/activate flamby

#MNIST
# seed=(0)
# momentum=(0.9)
# lr=(0.2)
# g_norm=(0.01 0.1)
# eps=(0.1 0.3 0.5)
# iidflag=("--save_dir=result")
# kf=("--save_dir=result")



# FLamby
# seed=(0) # 5 9 15)
# lr=(0.1 0.5)
# g_norm=(0.05 0.1 0.2)
# # eps=(0.3 0.5 1)
# eps=(0.5)
# kf=("--kf=True") # "--save_dir=result")
# iidflag=("--save_dir=result" "--noniid=True")

#CIFAR10
seed=(0)
sample_ratio=(0.1 1)
momentum=(0.0)
lr=(0.2 2)
g_norm=(0.01 0.1)
eps=(1 3)
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
            for gn in ${g_norm[@]}
            do
                # py_req="python ${cur_path}/main_lenet5.py --grad_norm=${gn} --local_round=2 --global_round=200 --lr=${l} --dataset=CIFAR10 --model=lenet5 ${k} --momentum=${m}";
                py_req="python ${cur_path}/main_stand.py --grad_norm=${gn} --local_round=2 --global_round=50 --lr=${l} --dataset=MNIST --model=cnn --momentum=${m} --sample_ratio=${r} --nium_clients=1000";
                # py_req="python ${cur_path}/main_flamby.py --seed=${s} --grad_norm=${gn} --local_round=2 --global_round=50 --lr=${l} --batch_size=2 --dataset=FLamby --model=mclr ${k}";                
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

output=0
echo "====DP====">>$logfile
for m in ${momentum[@]}
do
    for r in ${sample_ratio[@]}
    do
        for e in ${eps[@]}
        do
            for l in ${lr[@]}
            do
                for gn in ${g_norm[@]}
                do
                    # py_req="python ${cur_path}/main_stand.py --grad_norm=${gn} --dp=True --eps=${e}  --local_round=2 --global_round=200 --lr=${l} --dataset=CIFAR10 --model=lenet5 ${k} --momentum=${m}";
                    py_req="python ${cur_path}/main_fed.py --grad_norm=${gn} --dp=True --eps=${e}  --local_round=2 --global_round=50 --lr=${l} --dataset=MNIST --model=cnn  ${k} --momentum=${m}  --sample_ratio=${r} --num_clients=1000";
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