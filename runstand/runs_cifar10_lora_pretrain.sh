#!/bin/bash
set -o pipefail

data="CIFAR10"
dir_path=$(dirname $(pwd))
cur_date="`date +%Y%m%d`"

logfile_path=${dir_path}/logs/standalone
ckpt_path=${dir_path}/checkpoints/cifar10_cnn5_pretrain_40k_seed0.pt
logfile=${logfile_path}/log_cifar10_lora_pretrain_$cur_date
mkdir -p "$logfile_path"
mkdir -p "${dir_path}/checkpoints"

source /Users/liu/miniconda3/bin/activate venv
export PYTHONUNBUFFERED=1

echo "$(date '+%Y-%m-%d %H:%M:%S')" | tee -a "$logfile"
echo "====CIFAR10 CNN5 pretrain for LoRA split 40k====" | tee -a "$logfile"

py_req="python -u ${dir_path}/main_lora_stand.py --dataset=${data} --model=cnn5 --lora_split=pretrain --private_size=10000 --split_seed=0 --dp= --DR= --global_round=20 --local_round=20 --lr=0.1 --opt=sgd --num_clients=1 --batch_size=256 --save_checkpoint=${ckpt_path}"
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
echo "[time] pretrain py time is $(($cost_time/60))min $(($cost_time%60))s" | tee -a "$logfile"
echo "[finished]!" | tee -a "$logfile"
