#!/bin/bash
#SBATCH --job-name=drv3_stand
#SBATCH --output=out_drv3_stand
#SBATCH --gres=gpu:1
#SBATCH --mem=32GB
dir_path=$(dirname $(pwd))
echo "${dir_path}"
cur_date="`date +%Y%m%d`" 
data="SVHN"

logfile_path=${dir_path}/logs/
logfile=${dir_path}/logs/standalone/log_drv3_${data}_$cur_date
if [ ! -x $logfile_path ]; then
 mkdir "$logfile_path"
fi
source /local/scratch/yliu270/anaconda3/bin/activate flamby
if [ ! -f "$logfile" ]; then
 touch "$logfile"
fi

round=20
momentum=0.0
seed=(0)
lr=(2 4)
g_p_norm=(100)
clip_paral=(100) # alpha actucally
iidflag=("--save_dir=result")
opt=('sgd')

time=$(date "+%Y-%m-%d %H:%M:%S")
echo "${time}">>$logfile

echo "====DRtest=True; V3 ====">>$logfile

echo "====NoDP====">>$logfile
for o in ${opt[@]}
do
    for cp in ${clip_paral[@]}
    do
        for l in ${lr[@]}
        do
            for gpn in ${g_p_norm[@]}
            do
                # py_req="python ${dir_path}/main_stand.py --DRtest=True --grad_norm=100 --grad_perp_norm=${gpn} --local_round=${round} --global_round=${round} --lr=${l} --dataset=CIFAR10 --model=cnn5  --clip_paral=${cp} ${k} --opt=${o} --num_clients=1 --momentum=${momentum} --batch_size=256";
                # py_req="python ${dir_path}/main_stand.py --DRtest=True --grad_norm=100 --grad_perp_norm=${gpn} --local_round=${round} --global_round=${round} --lr=${l} --dataset=${data} --model=mclr  --clip_paral=${cp} ${k} --opt=${o} --num_clients=1 --momentum=${momentum} --batch_size=32";
                # py_req="python ${dir_path}/main_test.py --DRtest=True --grad_perp_norm=${gpn} --local_round=2 --global_round=100 --lr=${l} --dataset=MNIST --model=cnn  --clip_paral=${cp}";
                # py_req="python ${dir_path}/main_flamby.py --seed=${s} --DRtest=True --grad_perp_norm=${gpn} --local_round=2 --global_round=50 --lr=${l} --batch_size=2 --dataset=FLamby --model=mclr --clip_p=0.001";
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
lr=(2)
g_p_norm=(0.2)
clip_paral=(0.05) # alpha actucally
index=(0)
iidflag=("--save_dir=result")
opt=('sgd')

output=0
echo "====DP====">>$logfile
for m in ${momentum[@]}
do
    for s in ${seed[@]}
    do
        for cp in ${clip_paral[@]}
        do
            for l in ${lr[@]}
            do
                for gpn in ${g_p_norm[@]}
                do
                    # py_req="python ${dir_path}/main_stand.py --DRtest=True --seed=${s} --grad_norm=5 --grad_perp_norm=${gpn} --dp=True --eps=3  --local_round=${round} --global_round=${round} --lr=${l}  --dataset=CIFAR10 --clip_paral=${cp} --num_clients=1 --momentum=${m} --batch_size=256  --noise_multiplier_g=0.835 --noise_multiplier_p=0.84 --noise_multiplier_a=3.0";
                    py_req="python ${dir_path}/main_stand.py --DRtest=True --seed=${s} --grad_norm=5 --grad_perp_norm=${gpn} --dp=True --eps=3  --local_round=${round} --global_round=${round} --lr=${l}  --dataset=SVHN --clip_paral=${cp} --num_clients=1 --momentum=${m} --batch_size=256  --noise_multiplier_g=0.775 --noise_multiplier_p=0.78 --noise_multiplier_a=2.0";
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