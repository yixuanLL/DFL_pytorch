import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import make_interp_spline
import os
# import chardet

root_path = '/local/scratch/yliu270/workspace/DFL_pytorch/pics/res/'
file_list = os.listdir(root_path)
print(root_path)

ls = ['-', '-', '--', ':', '-.', '--', '--']
# m = ['o', '+', '*', ',', '^', 's', '*']
c = ['r', 'orange', 'slategrey', 'dodgerblue', 'blueviolet', 'darkcyan', 'yellowgreen']
plt.switch_backend('agg')

for file in file_list:
    file_path=root_path + file
    # with open(file_path, 'rb') as f:
    #     result = chardet.detect(f.read())  # or readline if the file is large
    df = pd.read_csv(file_path, encoding='utf-8')
    num_rounds = len(df.iloc[-1])
    acc = []
    dataset = df['data'][0]
    eps_dpsgd = df['epsilon_dpsgd']
    eps_dr = df['epsilon_dr']
    acc.extend((df['DPDRI'], df['DPDR'], df['DIFF'], df['DIFF2'], df['AutoClip'], df['DPAdam'], df['DPSGD']))
    exp=['DPDRI', 'DPDR', 'DIFF', 'DIFF2', 'AutoClip', 'DPAdam', 'DPSGD']
    print(file, dataset)


    for i in range(len(acc)):
        if 'DPDR' in exp[i]:
            eps = eps_dr
        else:
            eps = eps_dpsgd
        # print(eps)
        # plt.plot(eps, acc[i],  label=exp[i], linestyle=ls[i], marker=m[i], color=c[i])
        plt.plot(eps, acc[i],  label=exp[i], linestyle=ls[i], color=c[i])

    # 设置xy坐标范围
    # plt.ylim((10**1,10**5))
    # plt.xlim((.5,.8))
    
    #xy描述
    plt.ylabel('Accuracy')
    plt.xlabel('$\epsilon$')
    plt.legend(loc='lower right', fontsize=8)

    title = dataset + ' $\epsilon$=3'
    plt.title(title, fontsize=9)
    plt.show()
    file_name = 'conv_'+dataset+'.pdf'
    plt.savefig(root_path+file_name, dpi=600)
    plt.close()
    # f.close()