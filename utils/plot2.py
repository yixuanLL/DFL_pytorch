import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import make_interp_spline
import os

acc = []
round = []
acc_list=[]
exp_list=[]
round_list =[]
#read all the files
file_path = '/home/yliu270/workspace/DFL_pytorch/logs/tmp.txt'
root_path = '/home/yliu270/workspace/DFL_pytorch/logs/'
file = open(file_path, 'r')
for line in file:
    line = line.strip()
    if 'csv' in line:
        print(line)
        line = line.split('/')
        param = line[-1].split('-')
        l = param[0]
        e = line[-2]
        m = param[2]
        grad_norm = param[3]
        grad_p_norm = param[4]
    
    if 'round' in line:
        acc.append(float(line.split(' ')[7])*100)
        round.append(int(line.split(' ')[1]))
        if 'round 49' in line:
            acc_list.append(acc)
            exp_list.append(m)
            round_list.append(round)
            acc = []
            round = []



color = ['-.','b-.', 'g-.', 'c-',  'r', 'gold', 'orange', 'y-', 'b']
plt.switch_backend('agg')

for i in range(len(acc_list)):
    # print(round_list[i], acc_list[i], color[i], label=exp_list[i])
    plt.plot(round_list[i], acc_list[i], color[i], label=exp_list[i])


ls = ['--', ':', '-.', '--', '-', '-', '-']
m = ['o', '+', '*', ',', '^', 's', '*']
c = ['slategrey', 'dodgerblue', 'blueviolet', 'darkcyan', 'yellowgreen', 'r', 'orange']


# 设置xy坐标范围
# plt.ylim((10**1,10**5))
# plt.xlim((.5,.8))
 
#xy描述
plt.ylabel('Accuracy$')
plt.xlabel('Rounds')
plt.legend(loc='lower right', fontsize=8)

plt.title('MNIST $\epsilon$=0.5', fontsize=9)
plt.show()
plt.savefig(root_path+'tmp.png', dpi=600)
plt.close()