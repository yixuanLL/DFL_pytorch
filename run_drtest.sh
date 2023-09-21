#!/bin/bash
cur_path=`pwd`

cur_date="`date +%Y%m%d`" 

logfile_path=${cur_path}/logs/
logfile=${cur_path}/logs/log_drtest_$cur_date
if [ ! -x $logfile_path ]; then
 mkdir "$logfile_path"
fi

if [ ! -f "$logfile" ]; then
 touch "$logfile"
fi


# index=(0 1 2 3)
# lr=(0.05 0.1 0.5)
# g_p_norm=(1 0.5 0.1 0.01)
# eps=(1.9 0.4)
momentum=(0.0)
lr=(0.01)
g_p_norm=(1.0)
eps=(0.3 0.5 1 10e2)
eps_2=(0.05 0.1 10e6)



time=$(date "+%Y-%m-%d %H:%M:%S")
echo "${time}">>$logfile

echo "====NoDP====">>$logfile
for m in ${momentum[@]}
do
    for l in ${lr[@]}
    do
        for gpn in ${g_p_norm[@]}
        do
            py_req="python ${cur_path}/main_test.py --DRtest=True --grad_perp_norm=${gpn} --local_round=2 --global_round=200 --lr=${l} --dataset=CIFAR10 --model=lenet5 --momentum=${m}";
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


output=0
echo "====DP====">>$logfile
for e2 in ${eps_2[@]}
do
    for e in ${eps[@]}
    do
        for l in ${lr[@]}
        do
            for gpn in ${g_p_norm[@]}
            do
                py_req="python ${cur_path}/main_test.py --DRtest=True --eps=${e} --grad_perp_norm=${gpn} --dp=True --local_round=2 --global_round=100 --lr=${l} --dataset=MNIST --model=cnn  --eps_2=${e2}";
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


time=$(date "+%Y-%m-%d %H:%M:%S")
echo "${time}">>$logfile
echo "[finished]!"
echo "[finished]!">>$logfile