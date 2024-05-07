#!/bin/bash
#SBATCH --job-name=drv5_stand
#SBATCH --output=out_drv5_stand
#SBATCH --gres=gpu:1
#SBATCH --mem=24GB
data="CIFAR10"

dir_path=$(dirname $(pwd))
echo "${dir_path}"
cur_date="`date +%Y%m%d`" 

logfile_path=${dir_path}/logs/
logfile=${dir_path}/logs/standalone/log_drv5_${data}_$cur_date
if [ ! -x $logfile_path ]; then
 mkdir "$logfile_path"
fi
source /local/scratch/yliu270/anaconda3/bin/activate flamby
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
g_p_norm=(0.8)
clip_paral=(0.05) # alpha actucally

iidflag=("--save_dir=result")
opt=('sgd')
index=(2)
batch=(64 256 1024 4096)
noise_multiplier_g=(0.666 0.84 1.23 2.14)
noise_multiplier_p=(0.667 0.84 1.24 2.2)
noise_multiplier_a=(2 3 6 8)


output=0
echo "====DP: first 50 steps use SGD with different norm====">>$logfile
for id in ${index[@]}
do
    for s in ${seed[@]}
    do
        for cp in ${clip_paral[@]}
        do
            for l in ${lr[@]}
            do
                for gpn in ${g_p_norm[@]}
                do
                    py_req="python ${dir_path}/main_stand.py --seed=${s} --DR=True --grad_norm=5 --grad_perp_norm=${gpn} --dp=True --eps=3  --local_round=${round} --global_round=${round} --lr=${l}  --dataset=CIFAR10 --clip_paral=${cp} --num_clients=1 --batch_size=${batch[${id}]}  --noise_multiplier_g=${noise_multiplier_g[${id}]} --noise_multiplier_p=${noise_multiplier_p[${id}]} --noise_multiplier_a=${noise_multiplier_a[${id}]}";

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