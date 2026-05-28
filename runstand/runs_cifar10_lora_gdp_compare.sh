#!/bin/bash
set -o pipefail

data="CIFAR10"
dir_path=$(dirname $(pwd))
cur_date="`date +%Y%m%d`"

logfile_path=${dir_path}/logs/standalone
ckpt_path=${dir_path}/checkpoints/cifar10_cnn5_pretrain_40k_seed0.pt
logfile=${logfile_path}/log_cifar10_lora_gdp_compare_$cur_date
mkdir -p "$logfile_path"

source /Users/liu/miniconda3/bin/activate venv
export PYTHONUNBUFFERED=1
export PYTHONPATH="${dir_path}:${PYTHONPATH}"

echo "$(date '+%Y-%m-%d %H:%M:%S')" | tee -a "$logfile"
echo "====CIFAR10 LoRA GDP compare private 10k====" | tee -a "$logfile"

round=10
seed=0
lr=0.5
eps=3
delta=1e-5
batch=128
data_size=10000
steps_dr=800
steps_interval=40000

# GDP-calibrated for N=10000, batch=128, epochs=10, epsilon=3, delta=1e-5.
# Equal per-step privacy: 1/sigma_g^2 = 1/sigma_p^2 + 1/sigma_a^2.
noise_multiplier_g=0.786338
noise_multiplier_p=0.792232
noise_multiplier_a=6.457354

if [ ! -f "$ckpt_path" ]; then
    echo "[FAILED] missing pretrained checkpoint: $ckpt_path" | tee -a "$logfile"
    echo "Run runstand/runs_cifar10_lora_pretrain.sh first." | tee -a "$logfile"
    exit 8
fi

echo "====LoRA DPSGD GDP====" | tee -a "$logfile"
gdp_req="python -c '<DPSGD GDP accountant>' ${data_size} ${batch} ${round} ${delta} ${noise_multiplier_g}"
echo "${gdp_req}" | tee -a "$logfile"
python -c "import sys; from utils.dpsgd_utils import compute_dpsgd_gdp_mu, gdp_epsilon_from_mu; data_size=int(sys.argv[1]); batch_size=int(sys.argv[2]); epochs=int(sys.argv[3]); delta=float(sys.argv[4]); sigma_g=float(sys.argv[5]); mu=compute_dpsgd_gdp_mu(data_size,batch_size,epochs,sigma_g); eps=gdp_epsilon_from_mu(mu,delta); total_steps=int(data_size/batch_size*epochs); q=batch_size/data_size; print('CIFAR10-LoRA: DPSGD GDP epsilon=%.6f, mu=%.6f, q=%.6f, T=%d, sigma_g=%.6f' % (eps, mu, q, total_steps, sigma_g))" ${data_size} ${batch} ${round} ${delta} ${noise_multiplier_g} | tee -a "$logfile"

py_req="python -u ${dir_path}/main_lora_stand.py --dataset=${data} --model=cnn5_lora --lora_split=private --private_size=10000 --split_seed=0 --pretrained_path=${ckpt_path} --dp=True --DR= --grad_norm=0.2 --eps=${eps} --delta=${delta} --global_round=${round} --local_round=${round} --lr=${lr} --opt=sgd --num_clients=1 --batch_size=${batch} --noise_multiplier_g=${noise_multiplier_g}"
echo "${py_req}" | tee -a "$logfile"
start_time=$(date +%s)
${py_req} 2>&1 | tee -a "$logfile"
status=${PIPESTATUS[0]}
end_time=$(date +%s)
if [ $status -ne 0 ]; then
    echo "[FAILED] ${py_req}" | tee -a "$logfile"
    exit 8
fi
cost_time=$[ $end_time-$start_time ]
echo "[time] LoRA DPSGD py time is $(($cost_time/60))min $(($cost_time%60))s" | tee -a "$logfile"

echo "====LoRA DPDR GDP equal per-step privacy steps_dr=${steps_dr}====" | tee -a "$logfile"
gdp_req="python -c '<DPDR GDP accountant>' ${data_size} ${batch} ${round} ${delta} ${noise_multiplier_p} ${noise_multiplier_a} ${noise_multiplier_g} ${steps_dr} ${steps_interval}"
echo "${gdp_req}" | tee -a "$logfile"
python -c "import sys; from utils.dpsgd_utils import compute_dpdr_gdp_mu, gdp_epsilon_from_mu; data_size=int(sys.argv[1]); batch_size=int(sys.argv[2]); epochs=int(sys.argv[3]); delta=float(sys.argv[4]); sigma_p=float(sys.argv[5]); sigma_a=float(sys.argv[6]); sigma_g=float(sys.argv[7]); steps_dr=int(sys.argv[8]); steps_interval=int(sys.argv[9]); d=compute_dpdr_gdp_mu(data_size,batch_size,epochs,sigma_p,sigma_a,sigma_g,steps_dr,steps_interval,True); eps=gdp_epsilon_from_mu(d['mu'],delta); print('CIFAR10-LoRA: DPDR GDP epsilon=%.6f, mu=%.6f, q=%.6f, T=%d, T_gdr=%d, T_sgd=%d, sigma_p=%.6f, sigma_a=%.6f, sigma_g=%.6f' % (eps, d['mu'], d['q'], d['total_steps'], d['gdr_steps'], d['sgd_steps'], sigma_p, sigma_a, sigma_g))" ${data_size} ${batch} ${round} ${delta} ${noise_multiplier_p} ${noise_multiplier_a} ${noise_multiplier_g} ${steps_dr} ${steps_interval} | tee -a "$logfile"

py_req="python -u ${dir_path}/main_lora_stand.py --dataset=${data} --model=cnn5_lora --lora_split=private --private_size=10000 --split_seed=0 --pretrained_path=${ckpt_path} --dp=True --DR=True --grad_norm=5 --grad_perp_norm=0.2 --clip_paral=0.05 --eps=${eps} --delta=${delta} --global_round=${round} --local_round=${round} --lr=${lr} --opt=sgd --num_clients=1 --batch_size=${batch} --noise_multiplier_g=${noise_multiplier_g} --noise_multiplier_p=${noise_multiplier_p} --noise_multiplier_a=${noise_multiplier_a} --steps_dr=${steps_dr} --steps_interval=${steps_interval}"
echo "${py_req}" | tee -a "$logfile"
start_time=$(date +%s)
${py_req} 2>&1 | tee -a "$logfile"
status=${PIPESTATUS[0]}
end_time=$(date +%s)
if [ $status -ne 0 ]; then
    echo "[FAILED] ${py_req}" | tee -a "$logfile"
    exit 8
fi
cost_time=$[ $end_time-$start_time ]
echo "[time] LoRA DPDR py time is $(($cost_time/60))min $(($cost_time%60))s" | tee -a "$logfile"

echo "[finished]!" | tee -a "$logfile"
