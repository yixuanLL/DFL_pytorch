#!/bin/bash
#SBATCH --job-name=diff2_cifar_stand
#SBATCH --output=out_diff2_cifar_stand
#SBATCH --gres=gpu:1
#SBATCH --mem=8GB
dir_path=$(dirname $(pwd))
echo "${dir_path}"
cur_date="`date +%Y%m%d`" 

logfile_path=${dir_path}/logs/
logfile=${dir_path}/logs/standalone/log_diff2_cifar_$cur_date
if [ ! -x $logfile_path ]; then
 mkdir "$logfile_path"
fi

if [ ! -f "$logfile" ]; then
 touch "$logfile"
fi


# # MNIST
# seed=(0)
# lr=(0.5)
# g_p_norm=(0.01 0.05 0.1)
# eps=(0.04 0.06 0.08 0.1 0.12 0.32 0.52)
# # kf=("--kf=True")
# kf=("--save_dir=result") 
# iidflag=("--save_dir=result")
# # momentum=(0.0)




#FLamby
# seed=(0) #(5 9 15)
# lr=(0.1 0.5)
# g_p_norm=(0.05 0.1)
# eps=(0.3 0.5 1)
# # kf=("--save_dir=result") 
# kf=("--kf=True")
# iidflag=("--save_dir=result") # "--noniid=True")

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

echo "====DRtest=True; V6 ====">>$logfile

echo "====NoDP====">>$logfile
for o in ${opt[@]}
do
    for m in ${momentum[@]}
    do
        for l in ${lr[@]}
        do
            for gpn in ${g_p_norm[@]}
            do
                py_req="python ${dir_path}/main_diff2_stand.py --DRV2=True --grad_norm=100 --grad_perp_norm=${gpn} --local_round=${round} --global_round=${round} --lr=${l} --dataset=CIFAR10 --model=cnn5  ${k} --opt=${o} --num_clients=1 --momentum=${m} --batch_size=256";
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
momentum=(0.0 0.9)
seed=(0)
lr=(1 2 4)
g_p_norm=(0.05 0.1 0.5)
eps=(1 3)
iidflag=("--save_dir=result")
opt=('sgd')

output=0
echo "====DP====">>$logfile
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
                    py_req="python ${dir_path}/main_diff2_stand.py --DRV2=True --eps=${e} --grad_perp_norm=${gpn} --dp=True --local_round=${round} --global_round=${round} --lr=${l} --dataset=CIFAR10 --model=cnn5  --opt=${o} --num_clients=1 --momentum=${m} --batch_size=256";
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