#!/bin/bash
#SBATCH --job-name=drV3_cifar
#SBATCH --output=out_drV3_cifar
#SBATCH --gres=gpu:1
#SBATCH --mem=8GB
cur_path=`pwd`

cur_date="`date +%Y%m%d`" 

logfile_path=${cur_path}/logs/
logfile=${cur_path}/logs/log_drv3_cifar_$cur_date
if [ ! -x $logfile_path ]; then
 mkdir "$logfile_path"
fi

if [ ! -f "$logfile" ]; then
 touch "$logfile"
fi


# # MNIST
# seed=(0)
# lr=(0.1 0.2 0.5)
# g_p_norm=(0.01 0.1 0.5)
# clip_paral=(0.05 0.2 1)
# eps=(0.1 0.5 1)
# eps_2=(0.02 0.1 0.2)
# kf=("--kf=True")
# kf=("--save_dir=result") 
# iidflag=("--save_dir=result")
# momentum=(0.0)



#FLamby
seed=(0) #(5 9 15)
lr=(0.5)
g_p_norm=(0.01 0.05 0.1)
clip_paral=(0.01 0.1)
index=(1 2 3)
eps=(0.98 2.95)
eps_2=(0.02 0.05)
kf=("--save_dir=result") 
# kf=("--kf=True")
iidflag=("--save_dir=result") # "--noniid=True")

#CIFAR10
# seed=(0)
# momentum=(0.0)
# lr=(0.2 0.5 1)
# g_p_norm=(0.01 0.1 1.0)
# clip_paral=(0.2)
# eps=(0.1 0.5 1)
# eps_2=(0.05 0.1)
# iidflag=("--save_dir=result")

time=$(date "+%Y-%m-%d %H:%M:%S")
echo "${time}">>$logfile

echo "====DRtest=True; V3 ====">>$logfile

echo "====NoDP====">>$logfile
for s in ${seed[@]}
do
    for cp in ${clip_paral[@]}
    do
        for l in ${lr[@]}
        do
            for gpn in ${g_p_norm[@]}
            do
                # py_req="python ${cur_path}/main_test.py --DR=True --grad_perp_norm=${gpn} --local_round=2 --global_round=200 --lr=${l} --dataset=CIFAR10 --model=lenet5 --clip_paral=${cp}";
                # py_req="python ${cur_path}/main_test.py --DR=True --grad_perp_norm=${gpn} --local_round=2 --global_round=100 --lr=${l} --dataset=MNIST --model=cnn  --clip_paral=${cp}";
                py_req="python ${cur_path}/main_test.py --seed=${s} --DR=True --grad_perp_norm=${gpn} --local_round=2 --global_round=50 --lr=${l} --batch_size=16 --dataset=FLamby --model=mclr --clip_paral=${cp}";
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


output=0
echo "====DP====">>$logfile
for s in ${seed[@]}
do
    for i in ${index[@]}
    do
        for cp in ${clip_paral[@]}
        do
            for l in ${lr[@]}
            do
                for gpn in ${g_p_norm[@]}
                do
                    # py_req="python ${cur_path}/main_test.py --DR=True --eps=${e} --grad_perp_norm=${gpn} --dp=True --local_round=2 --global_round=200 --lr=${l} --dataset=CIFAR10 --model=lenet5 --eps_2=${e2} --clip_paral=${cp}";
                    # py_req="python ${cur_path}/main_test.py --DR=True --eps=${e} --grad_perp_norm=${gpn} --dp=True --local_round=2 --global_round=100 --lr=${l} --dataset=MNIST --model=cnn --eps_2=${e2} --clip_paral=${cp}";
                    py_req="python ${cur_path}/main_test.py --seed=${s} --DR=True --dp=True --eps=${e} --grad_perp_norm=${gpn} --local_round=2 --global_round=50 --lr=${l} --batch_size=16 --eps_2=${e2} --clip_paral=${cp} --dataset=FLamby --model=mclr";
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