#!/bin/bash

data="MNIST"

dir_path=$(dirname $(pwd))
echo "${dir_path}"
cur_date="`date +%Y%m%d`" 

logfile_path=${dir_path}/logs/standalone
logfile=${logfile_path}/log_dpsgd_gdp_${data}_$cur_date
if [ ! -x $logfile_path ]; then
 mkdir -p "$logfile_path"
fi
# source /local/scratch/yliu270/anaconda3/bin/activate flamby
source /Users/liu/miniconda3/bin/activate venv
if [ ! -f "$logfile" ]; then
 touch "$logfile"
fi

time=$(date "+%Y-%m-%d %H:%M:%S")
echo "${time}">>$logfile

round=20
seed=(0)
lr=(2)
g_norm=(0.2)
eps=3
delta=1e-5
opt=('sgd')
batch=(256)
noise_multiplier_g=(0.715281)

if [ "$data" = "MNIST" ]; then
    data_size=60000
elif [ "$data" = "CIFAR10" ]; then
    data_size=50000
elif [ "$data" = "SVHN" ]; then
    data_size=73257
else
    echo "[FAILED] Unknown dataset size for ${data}"
    echo "[FAILED] Unknown dataset size for ${data}">>$logfile
    exit 8
fi

echo "====DPSGD GDP====">>$logfile
for l in ${lr[@]}
do
    for o in ${opt[@]}
    do
        for s in ${seed[@]}
        do
            for gn in ${g_norm[@]}
            do
                for id in ${!batch[@]}
                do
                    b=${batch[${id}]}
                    nm=${noise_multiplier_g[${id}]}

                    gdp_req="python -c '<GDP accountant>' ${data_size} ${b} ${round} ${delta} ${nm} ${data} ${dir_path}"
                    echo "${gdp_req}"
                    echo "${gdp_req}">>$logfile
                    gdp_output=`python -c "import sys; sys.path.insert(0, sys.argv[7]); from utils.dpsgd_utils import compute_dpsgd_gdp_mu, gdp_epsilon_from_mu; data_size=int(sys.argv[1]); batch_size=int(sys.argv[2]); epochs=int(sys.argv[3]); delta=float(sys.argv[4]); sigma_g=float(sys.argv[5]); data=sys.argv[6]; mu=compute_dpsgd_gdp_mu(data_size, batch_size, epochs, sigma_g); eps=gdp_epsilon_from_mu(mu, delta); total_steps=int(data_size / batch_size * epochs); q=batch_size / data_size; print('%s: DPSGD GDP epsilon=%.6f, mu=%.6f, q=%.6f, T=%d, sigma_g=%.6f' % (data, eps, mu, q, total_steps, sigma_g))" ${data_size} ${b} ${round} ${delta} ${nm} ${data} ${dir_path}`
                    if [ $? -ne 0 ]; then
                        echo "[FAILED] ${gdp_req}"
                        echo "[FAILED] ${gdp_req}">>$logfile
                        exit 8
                    fi
                    echo "${gdp_output}"
                    echo "${gdp_output}">>$logfile

                    py_req="python ${dir_path}/main_stand.py --seed=${s} --DR= --grad_norm=${gn} --dp=True --eps=${eps} --local_round=${round} --global_round=${round} --lr=${l} --dataset=${data} --opt=${o} --num_clients=1 --batch_size=${b} --noise_multiplier_g=${nm}";
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
