#!/bin/bash
#SBATCH --job-name=drV4
#SBATCH --output=out_drV4
#SBATCH --gres=gpu:1
#SBATCH --mem=8GB
cur_path=`pwd`

cur_date="`date +%Y%m%d`" 

logfile_path=${cur_path}/logs/
logfile=${cur_path}/logs/log_drv4_$cur_date
if [ ! -x $logfile_path ]; then
 mkdir "$logfile_path"
fi

if [ ! -f "$logfile" ]; then
 touch "$logfile"
fi


# # MNIST
seed=(0)
lr=(0.5)
g_p_norm=(0.01 0.05)
clip_paral=(0.2)
eps=(0.1 0.3 0.5)
eps_2=(0.05 0.1)
# kf=("--kf=True")
kf=("--save_dir=result") 
iidflag=("--save_dir=result")
# momentum=(0.0)




#FLamby
# seed=(0) #(5 9 15)
# lr=(0.1 0.5)
# g_p_norm=(0.05 0.1)
# eps=(0.3 0.5 1)
# # kf=("--save_dir=result") 
# kf=("--kf=True")
# iidflag=("--save_dir=result") # "--noniid=True")

#CIFAR10
# seed=(0)
# momentum=(0.0)
# lr=(0.1 0.5)
# g_p_norm=(0.1 0.3 1.0)
# eps=(0.3 0.5 1)
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
                # py_req="python ${cur_path}/main_test.py --DRtest=True --grad_perp_norm=${gpn} --local_round=2 --global_round=200 --lr=${l} --dataset=CIFAR10 --model=lenet5 --clip_paral=0.05";
                py_req="python ${cur_path}/main_test.py --DRtest=True --grad_perp_norm=${gpn} --local_round=2 --global_round=100 --lr=${l} --dataset=MNIST --model=cnn  --clip_paral=${cp}";
                # py_req="python ${cur_path}/main_flamby.py --seed=${s} --DRtest=True --grad_perp_norm=${gpn} --local_round=2 --global_round=50 --lr=${l} --batch_size=2 --dataset=FLamby --model=mclr --clip_p=0.001";
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
    for cp in ${clip_paral[@]}
    do
        for e2 in ${eps_2[@]}
        do
            for e in ${eps[@]}
            do
                for l in ${lr[@]}
                do
                    for gpn in ${g_p_norm[@]}
                    do
                        # py_req="python ${cur_path}/main_test.py --DRtest=True --dp=True --eps=${e} --eps_2=0.05 --grad_perp_norm=${gpn} --local_round=2 --global_round=200 --lr=${l} --dataset=CIFAR10 --model=lenet5 --clip_paral=0.05 ${k}";
                        py_req="python ${cur_path}/main_test.py --DRtest=True --eps=${e} --grad_perp_norm=${gpn} --dp=True --local_round=2 --global_round=100 --lr=${l} --dataset=MNIST --model=cnn --eps_2=${e2} --clip_paral=${cp}";
                        # py_req="python ${cur_path}/main_flamby.py --seed=${s} --DRtest=True --dp=True --eps=${e} --grad_perp_norm=${gpn} --local_round=2 --global_round=50 --lr=${l} --batch_size=2 --eps_2=0.02 --dataset=FLamby --model=mclr --clip_p=0.001 ${k}";
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