#!/bin/bash
cur_path=`pwd`

cur_date="`date +%Y%m%d`" 

logfile_path=${cur_path}/logs/
logfile=${cur_path}/logs/log_sgd_nid_$cur_date
if [ ! -x $logfile_path ]; then
 mkdir "$logfile_path"
fi

if [ ! -f "$logfile" ]; then
 touch "$logfile"
fi


# eps=(2 1 0.5 0.3)
# g_norm=(10 5 2 1 0.5 0.1)

lr=(0.01 0.05 0.1 0.5)
g_norm=(1 0.5 0.1 0.01)
eps=(2 1 0.5 0.3)

# lr=(0.01)
# g_norm=(0.01)
# eps=(2 0.5)

time=$(date "+%Y-%m-%d %H:%M:%S")
echo "${time}">>$logfile

output=0
echo "====NoDP====">>$logfile
for l in ${lr[@]}
do
    for gn in ${g_norm[@]}
    do
        py_req="python ${cur_path}/main_lenet5.py --noniid=True --grad_norm=${gn} --dp=True --local_round=2 --global_round=200 --lr=${l}";
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

output=0
echo "====DP====">>$logfile
for e in ${eps[@]}
do
    for l in ${lr[@]}
    do
        for gn in ${g_norm[@]}
        do
            py_req="python ${cur_path}/main_lenet5.py --noniid=True --grad_norm=${gn} --dp=True --eps=${e}  --local_round=2 --global_round=200 --lr=${l}";
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

time=$(date "+%Y-%m-%d %H:%M:%S")
echo "${time}">>$logfile
echo "[finished]!"
echo "[finished]!">>$logfile