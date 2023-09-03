import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import make_interp_spline
import os

acc = []
exp = []
round = []
#read all the files
root_path = '/home/yixuan/workspace/DFL_pytorch/result/result/CIFAR10/iid/lenet5/'
folder_list = os.listdir(root_path)
for folder_eps in folder_list:
    # print(folder_list[10], folder_eps)
    
    if  folder_eps in ['no-dp','1.0','0.9']:
        eps_path = root_path + folder_eps
        file_list = os.listdir(eps_path)
        for file in file_list:
            file_path = eps_path + '/' + file
            name_list = file.strip('.csv').split('-')
            C = ''
            alg = ''
            if len(name_list) < 5:
                alg = 'SGD'
                C=name_list[1]
            else:
                alg = name_list[1]
                C=name_list[3]


            if folder_eps=='0.9':
                folder_eps = '1.0'
            # if alg in ['SGD', 'DR'] and (C in ['0.5', 'inf']):
            if alg in ['SGD', 'DR'] and (C in ['1.0', 'inf']):
                exp_name = alg + ', eps='+folder_eps+', C='+C


            # if flag:
                #读取csv文件
                df = pd.read_csv(file_path, header=None)
                num_rounds = len(df.iloc[-1])
                if alg=='DR' and folder_eps=='no-dp':
                    continue
                # if alg == 'SGD' and folder_eps=='no-dp' and C == '0.5':
                if alg == 'SGD' and folder_eps=='no-dp' and C == '1.0':
                    continue
                acc.append(df.iloc[-1])
                exp.append(exp_name)
                round.append([i for i in range(num_rounds)])

                print(file, exp_name)




color = ['-.','b-.', 'g-.', 'c-',  'r', 'gold', 'orange', 'y-', 'b']
plt.switch_backend('agg')

for i in range(len(acc)):
    plt.plot(round[i], acc[i], color[i], label=exp[i])


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

plt.title('LeNet5 CIFAR10 $\epsilon$=1.', fontsize=9)
plt.show()
plt.savefig(root_path+'acc_c1.png', dpi=600)
plt.close()