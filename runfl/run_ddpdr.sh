#!/bin/bash
#SBATCH --job-name=feddr
#SBATCH --output=out_feddr
#SBATCH --gres=gpu:1
#SBATCH --mem=8GB
cur_path=`pwd`

cur_date="`date +%Y%m%d`" 
dataset="MNIST"

logfile_path=${cur_path}/logs/
logfile=${cur_path}/logs/log_ddp_dr_${dataset}_$cur_date
if [ ! -x $logfile_path ]; then
 mkdir "$logfile_path"
fi

if [ ! -f "$logfile" ]; then
 touch "$logfile"
fi
source /home/yliu270/anaconda3/bin/activate flamby


seed=(0)
sample_ratio=(0.1)
momentum=(0.0)
lr=(0.1 0.5)
global_lr=(2 3)
clip_paral=(0.02 0.05)
gp_norm=(0.05 0.1 0.3)
index=(0 1)
eps=(2.9 2.95)
eps_2=(0.1 0.05)
# kf=("--kf=True")
kf=("--save_dir=result")
iidflag=("--save_dir=result")

time=$(date "+%Y-%m-%d %H:%M:%S")
echo "${time}">>$logfile

output=0
echo "====DR====">>$logfile
for cp in ${clip_paral[@]}
do
    for glr in ${global_lr[@]}
    do
        for r in ${sample_ratio[@]}
        do
            for i in ${index[@]}
            do
                for l in ${lr[@]}
                do
                    for gp in ${gp_norm[@]}
                    do
                        # py_req="python ${cur_path}/main_stand.py --grad_norm=${gn} --dp=True --eps=${e}  --local_round=2 --global_round=200 --lr=${l} --dataset=CIFAR10 --model=lenet5 ${k} --momentum=${m}";
                        py_req="python ${cur_path}/main_fed.py --grad_norm=0.3 --grad_perp_norm=${gp} --eps=${eps[${i}]} --eps_2=${eps_2[${i}]}  --local_round=1 --global_round=100 --lr=${l} --glr=${glr} --dataset=${dataset} --model=cnn  --clip_paral=${cp} --sample_ratio=${r} --num_clients=1000  --FLalg=FedDRDP";
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
done

time=$(date "+%Y-%m-%d %H:%M:%S")
echo "${time}">>$logfile
echo "[finished]!"
echo "[finished]!">>$logfile