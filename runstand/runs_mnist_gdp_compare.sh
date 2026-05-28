#!/bin/bash
set -o pipefail

data="MNIST"

dir_path=$(dirname $(pwd))
echo "${dir_path}"
cur_date="`date +%Y%m%d`"

logfile_path=${dir_path}/logs/standalone
logfile=${logfile_path}/log_mnist_gdp_compare_$cur_date
if [ ! -x $logfile_path ]; then
 mkdir -p "$logfile_path"
fi
# source /local/scratch/yliu270/anaconda3/bin/activate flamby
source /Users/liu/miniconda3/bin/activate venv
export PYTHONUNBUFFERED=1
if [ ! -f "$logfile" ]; then
 touch "$logfile"
fi

time=$(date "+%Y-%m-%d %H:%M:%S")
echo "${time}">>$logfile

round=20
seed=(0)
lr=(2)
eps=3
delta=1e-5
opt=('sgd')
batch=(256)
data_size=60000
steps_dr_list=(50 500)

echo "====MNIST DPDR GDP equal per-step privacy optimal sigma split====">>$logfile
for b in ${batch[@]}
do
    for s in ${seed[@]}
    do
        for l in ${lr[@]}
        do
            for o in ${opt[@]}
            do
                for steps_dr in ${steps_dr_list[@]}
                do
                    gpn=0.2
                    cp=0.05
                    steps_interval=40000
                    noise_multiplier_p=0.716847
                    noise_multiplier_a=10.826021
                    noise_multiplier_g=0.715280313

                    echo "====MNIST DPDR GDP equal per-step privacy steps_dr=${steps_dr}===="
                    echo "====MNIST DPDR GDP equal per-step privacy steps_dr=${steps_dr}====">>$logfile

                    gdp_req="python -c '<DPDR GDP accountant>' ${data_size} ${b} ${round} ${delta} ${noise_multiplier_p} ${noise_multiplier_a} ${noise_multiplier_g} ${steps_dr} ${steps_interval} ${data} ${dir_path}"
                    echo "${gdp_req}"
                    echo "${gdp_req}">>$logfile
                    gdp_output=`python -c "import sys; sys.path.insert(0, sys.argv[11]); from utils.dpsgd_utils import compute_dpdr_gdp_mu, gdp_epsilon_from_mu; data_size=int(sys.argv[1]); batch_size=int(sys.argv[2]); epochs=int(sys.argv[3]); delta=float(sys.argv[4]); sigma_p=float(sys.argv[5]); sigma_a=float(sys.argv[6]); sigma_g=float(sys.argv[7]); steps_dr=int(sys.argv[8]); steps_interval=int(sys.argv[9]); data=sys.argv[10]; d=compute_dpdr_gdp_mu(data_size,batch_size,epochs,sigma_p,sigma_a,sigma_g,steps_dr,steps_interval,True); eps=gdp_epsilon_from_mu(d['mu'], delta); print('%s: DPDR GDP epsilon=%.6f, mu=%.6f, q=%.6f, T=%d, T_gdr=%d, T_sgd=%d, sigma_p=%.6f, sigma_a=%.6f, sigma_g=%.6f' % (data, eps, d['mu'], d['q'], d['total_steps'], d['gdr_steps'], d['sgd_steps'], sigma_p, sigma_a, sigma_g))" ${data_size} ${b} ${round} ${delta} ${noise_multiplier_p} ${noise_multiplier_a} ${noise_multiplier_g} ${steps_dr} ${steps_interval} ${data} ${dir_path}`
                    if [ $? -ne 0 ]; then
                        echo "[FAILED] ${gdp_req}"
                        echo "[FAILED] ${gdp_req}">>$logfile
                        exit 8
                    fi
                    echo "${gdp_output}"
                    echo "${gdp_output}">>$logfile

                    py_req="python -u ${dir_path}/main_stand.py --seed=${s} --DR=True --grad_norm=5 --grad_perp_norm=${gpn} --dp=True --eps=${eps} --local_round=${round} --global_round=${round} --lr=${l} --dataset=${data} --clip_paral=${cp} --opt=${o} --num_clients=1 --batch_size=${b} --noise_multiplier_g=${noise_multiplier_g} --noise_multiplier_p=${noise_multiplier_p} --noise_multiplier_a=${noise_multiplier_a} --steps_dr=${steps_dr} --steps_interval=${steps_interval}";
                    echo "${py_req}"
                    echo "${py_req}">>$logfile
                    start_time=$(date +%s)
                    ${py_req} 2>&1 | tee -a "$logfile"
                    status=${PIPESTATUS[0]}
                    end_time=$(date +%s)
                    if [ $status -ne 0 ]; then
                        echo "[FAILED] ${py_req}"
                        echo "[FAILED] ${py_req}">>$logfile
                        exit 8
                    fi
                    sleep 1;
                    cost_time=$[ $end_time-$start_time ]
                    time=$(date "+%H:%M:%S")
                    echo "${time}">>$logfile
                    echo "[time] DPDR steps_dr=${steps_dr} build py time is $(($cost_time/60))min $(($cost_time%60))s"
                    echo "[time] DPDR steps_dr=${steps_dr} build py time is $(($cost_time/60))min $(($cost_time%60))s">>$logfile
                done
            done
        done
    done
done

time=$(date "+%Y-%m-%d %H:%M:%S")
echo "${time}">>$logfile
echo "[finished]!"
echo "[finished]!">>$logfile
