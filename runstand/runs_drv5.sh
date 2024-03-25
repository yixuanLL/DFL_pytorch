#!/bin/bash
#SBATCH --job-name=drv5_stand
#SBATCH --output=out_drv5_stand
#SBATCH --gres=gpu:1
#SBATCH --mem=24GB
data="SVHN"

dir_path=$(dirname $(pwd))
echo "${dir_path}"
cur_date="`date +%Y%m%d`" 

logfile_path=${dir_path}/logs/
logfile=${dir_path}/logs/standalone/log_drv5_${data}_$cur_date
if [ ! -x $logfile_path ]; then
 mkdir "$logfile_path"
fi

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
g_p_norm=(0.1 0.5 1 1.5 2)
clip_paral=(1 1.5) # alpha actucally
index=(0)
eps=(2.98)
eps_2=(0.02)
iidflag=("--save_dir=result")
opt=('sgd')
batch=(128)

output=0
echo "====DP: first 50 steps use SGD with different norm====">>$logfile
for b in ${batch[@]}
do
    for i in ${index[@]}
    do
        for cp in ${clip_paral[@]}
        do
            for l in ${lr[@]}
            do
                for gpn in ${g_p_norm[@]}
                do
                    py_req="python ${dir_path}/main_stand.py --DR=True --eps=${eps[${i}]} --eps_2=${eps_2[${i}]} --grad_norm=5 --grad_perp_norm=${gpn} --dp=True --local_round=${round} --global_round=${round} --lr=${l} --dataset=${data} --clip_paral=${cp}  --num_clients=1 --momentum=${momentum} --batch_size=${b}";
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