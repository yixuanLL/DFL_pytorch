#!/bin/bash
#SBATCH --job-name=diff2_stand
#SBATCH --output=out_diff2_stand
#SBATCH --gres=gpu:1
#SBATCH --mem=16GB
dir_path=$(dirname $(pwd))
echo "${dir_path}"
cur_date="`date +%Y%m%d`" 
data="CIFAR10"

logfile_path=${dir_path}/logs/
logfile=${dir_path}/logs/standalone/log_diff2_${data}_$cur_date
if [ ! -x $logfile_path ]; then
 mkdir "$logfile_path"
fi

if [ ! -f "$logfile" ]; then
 touch "$logfile"
fi

source /local/scratch/yliu270/anaconda3/bin/activate flamby

#CIFAR10
round=20
momentum=(0.0)
seed=(0)
lr=(0.1 1)
g_p_norm=(100)
iidflag=("--save_dir=result")
opt=('sgd')

time=$(date "+%Y-%m-%d %H:%M:%S")
echo "${time}">>$logfile

py_req='0'
echo "====NoDP====">>$logfile
for o in ${opt[@]}
do
    for m in ${momentum[@]}
    do
        for l in ${lr[@]}
        do
            for gpn in ${g_p_norm[@]}
            do
                # py_req="python ${dir_path}/main_diff2_stand.py --DRV2=True --grad_norm=100 --grad_perp_norm=${gpn} --local_round=${round} --global_round=${round} --lr=${l} --dataset=CIFAR10 --model=cnn5  ${k} --opt=${o} --num_clients=1 --momentum=${m} --batch_size=256";
                # py_req="python ${dir_path}/main_diff2_stand.py --DRV2=True --grad_norm=100 --grad_perp_norm=${gpn} --local_round=${round} --global_round=${round} --lr=${l} --dataset=CIFAR10 --opt=${o} --num_clients=1 --momentum=${m} --batch_size=${b}";
                echo "${py_req}"
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
round=20
momentum=(0.0)
seed=(0)
lr=(2 4)
g_p_norm=(0.01 0.05 0.1 0.3)
eps=(3)
iidflag=("--save_dir=result")
opt=('sgd')
batch=(256)


output=0
echo "====DP====">>$logfile
for b in ${batch[@]}
do
    for o in ${opt[@]}
    do
        for e in ${eps[@]}
        do
            for m in ${momentum[@]}
            do
                for l in ${lr[@]}
                do
                    for gpn in ${g_p_norm[@]}
                    do
                        # py_req="python ${dir_path}/main_diff2_stand.py --DRV2=True --eps=${e} --grad_norm=5 --grad_perp_norm=${gpn} --dp=True --local_round=${round} --global_round=${round} --lr=${l} --dataset=${data} --opt=${o} --num_clients=1 --momentum=${m} --batch_size=${b}";
                        py_req="python ${dir_path}/main_diff2_stand.py --DRV2=True --eps=${e} --seed=0 --grad_norm=5 --grad_perp_norm=${gpn} --dp=True --eps=3 --local_round=${round} --global_round=${round} --lr=${l} --dataset=CIFAR10  --opt=${o} --num_clients=1 --batch_size=${b} --noise_multiplier_g=0.835 --noise_multiplier_p=0.84 --noise_multiplier_a=3.0";
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