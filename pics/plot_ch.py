import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import make_interp_spline
import os
import torch
import pylab
# import chardet

# root_path = '/local/scratch/yliu270/workspace/DFL_pytorch/pics/'
root_path = '/home/liuyixuan/workspace/DFL_pytorch/pics/'


ls = ['-', '--', '--', '--', '--', '--', '--']
m = ['o', '+', '*', ',', '^', 's', '*']
c = ['r', 'g', 'orange', 'dodgerblue', 'blueviolet', 'darkcyan', 'yellowgreen']
plt.switch_backend('agg')
plt.rcParams['font.sans-serif'] = ['SimHei']

def convergence_plot():
    file_list_path = root_path + 'res_conv/'
    file_list = os.listdir(file_list_path)
    print(file_list_path)
    for file in file_list:
        if 'csv' not in file:
            continue
        file_path= file_list_path + file
        df = pd.read_csv(file_path, encoding='utf-8')
        num_rounds = len(df.iloc[-1])
        acc = []
        dataset = df['data'][0]
        eps_dpsgd = df['epsilon_dpsgd']
        eps_dr = df['epsilon_dr']
        acc.extend((df['DPDR'], df['DIFF'], df['DIFF2'], df['AutoClip'], df['DPAdam'], df['DPSGD']))
        exp=['DPDR', 'DIFF', 'DIFF2', 'AutoClip', 'DPAdam', 'DPSGD']
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
        plt.ylabel('准确率 ', fontsize=16)
        plt.xlabel('$\epsilon$', fontsize=16)
        plt.legend(loc='lower right', fontsize=16)

        title = dataset #+ ', $(\epsilon,\delta)=(3,10^{-5})$'
        plt.title(title, fontsize=16)
        plt.show()
        file_name = 'conv_'+dataset+'_ch.pdf'
        plt.savefig(file_list_path+file_name, dpi=600)
        plt.close()

def parameter_plot():
    ls = ['','--', '--', '--', ':', '-.', '--', '--']
    m = ['','^', '^', '*', ',', '^', 's', '*']
    c = ['','forestgreen', 'orange', 'slategrey', 'dodgerblue', 'blueviolet', 'darkcyan']
    file_list_path = root_path + 'res_param/'
    file_list = os.listdir(file_list_path)
    for file in file_list:
        if '.csv'  not in file:
            continue
        file_path=file_list_path + file    
        df = pd.read_csv(file_path, encoding='utf-8')
        head = df.columns.values
        param_name = head[0]
        print(param_name)
        acc = []
        params = df[param_name]
        for i in range(1, len(head)):
            exp=head[i]
            acc=df[exp]
            # print(acc)
            plt.plot(params, acc,  label=exp, linestyle=ls[i], marker=m[i],color=c[i])
        # acc.extend((df['DPDRI'], df['DPSGD']))
        # exp=['DPDRI', 'DPSGD']
        # for i in range(len(acc)):
            # plt.plot(params, acc[i],  label=exp[i], linestyle=ls[i], marker=m[i],color=c[i])

        # 设置xy坐标范围
        plt.ylim((40, 65))
        # plt.xlim((.5,.8))
        plt.xscale('log')
        
        #xy描述
        if 'Alpha' not in param_name:
            # param_name_title = param_name.replace('_',' ')
            if 'Batch' in param_name:
                param_name_title = "批次大小"
            if 'Steps' in param_name:
                param_name_title = "前期步数"
        else:
            # param_name_title = 'Alpha Bound $C_{\\alpha}$'
            param_name_title = '$C_{\\alpha}$'
            print(param_name_title)
        plt.ylabel('准确率', fontsize=16)
        plt.xlabel(param_name_title, fontsize=16)
        plt.legend(loc='lower right', fontsize=16)

        title = '参数：' + param_name_title #+ ', $(\epsilon,\delta)=(3,10^{-5})$'
        plt.title(title, fontsize=16)
        plt.show()
        file_name = 'param_'+param_name+'.pdf'
        plt.savefig(file_list_path+file_name, dpi=600)
        plt.close()

def segment_plot():
    ls = ['-','-', '-', ':', ':', '-.', '--', '--']
    m = ['','', '', '*', ',', '^', 's', '*']
    a = [0.8,1, 0.5]
    # c = ['orange', 'yellowgreen',  'dodgerblue', 'slategrey','blueviolet', 'darkcyan']
    # c = ['#36600E', '#78A040']
    c = ['#36600E', 'orange']
    file_list_path = root_path + 'res_grad/'
    file_list = os.listdir(file_list_path)
    # labels = ["grad_norm", "gperp_norm", "gdiff_norm","gpn2gn"]
    labels = ["grad_norm", "gperp_norm"]#, "gdiff_norm"]
    # label_name={"grad_norm":"entire grad", "gperp_norm":"orthogonal grad"}
    label_name={"grad_norm":"梯度", "gperp_norm":"梯度正交分量", "gdiff_norm":"梯度差值"}
    k=0
    for file in file_list:
        if "txt" not in file:
            continue
        print(file)
        file_path=file_list_path + file 
        try:   
            df = pd.read_csv(file_path,header=None, encoding='utf-8')
        except:
            continue
        res = {}
        exp_name = ' '.join((file.split('_')[0:2]))
        steps = 0
        for i in range(len(df)):
            data_line = df.iloc[i]
            round = data_line[0]
            label = data_line[1]
            value = data_line[3:].tolist()
            if label not in res:
                res[label] = value
            else:
                res[label].append(value)
        j=0
        bound = [0,0]
        for label in labels:
            # exp=exp_name + ' ' +label
            exp = label_name[label]
            y = res[label]
            steps = range(1,len(y)+1, 3)
            y=y[::3]
            bound[j] = y
            print(len(steps),len(y))
            # plt.plot(steps, y,  label=exp, linestyle=ls[j], marker=m[j],color=c[k],alpha=a[j])
            plt.plot(steps, y,  label=exp, linestyle=ls[j], marker=m[j],color=c[j],alpha=a[j])
            j+=1
        k += 1
        zero_bound = [0]*len(steps)
        plt.fill_between(steps, bound[0], bound[1],   alpha=0.8, label='平行', color='w', edgecolor=c[0], hatch='\\') 
        plt.fill_between(steps, bound[1], zero_bound,  alpha=0.1, label='正交', color=c[1]) #facecolor='C0',
    # 设置xy坐标范围
    # plt.ylim((40, 65))
    plt.xlim((0,250))
    
    # plt.xscale('log')
    
    #xy描述
    plt.ylabel('范数', fontsize=16)
    plt.xlabel("步数", fontsize=16)
    plt.legend(loc='upper right', fontsize=16)

    # title = 'Norm ratio' + param_name
    # plt.title(title, fontsize=9)
    plt.show()
    file_name = 'grad_early_common_knowledge_cn.pdf'
    plt.savefig(file_list_path+file_name, dpi=600)
    plt.close()

def diff_plot():
    ls = ['-','-', '-', ':', ':', '-.', '--', '--']
    m = ['','', '', '*', ',', '^', 's', '*']
    a = [1,1, 0.5]
    c = ['forestgreen', 'orange', 'forestgreen', 'forestgreen',  'blueviolet', 'dodgerblue', 'slategrey','blueviolet', 'darkcyan']
    # c = ['#36600E', '#78A040']
    # c = ['#36600E', 'orange']
    file_list_path = root_path + 'res_grad/'
    file_list = os.listdir(file_list_path)
    # labels = ["grad_norm", "gperp_norm", "gdiff_norm","gpn2gn"]
    labels = ["grad_norm", "gdiff_norm"]
    # label_name={"grad_norm":"entire grad", "gperp_norm":"orthogonal grad", "gdiff_norm":"grad difference"}
    label_name={"grad_norm":"梯度", "gperp_norm":"梯度正交分量", "gdiff_norm":"梯度差值"}
    
    k=0
    for file in file_list:
        if "txt" not in file:
            continue
        print(file)
        file_path=file_list_path + file 
        try:   
            df = pd.read_csv(file_path,header=None, encoding='utf-8')
        except:
            continue
        res = {}
        exp_name = ' '.join((file.split('_')[0:2]))
        steps = 0
        for i in range(len(df)):
            data_line = df.iloc[i]
            round = data_line[0]
            label = data_line[1]
            value = data_line[3:].tolist()
            if label not in res:
                res[label] = value
            else:
                res[label].append(value)
        j=0
        bound = [0,0]
        for label in labels:
            # exp=exp_name + ' ' +label
            exp = label_name[label]
            y = res[label]
            steps = range(1,len(y)+1, 3)
            y=y[::3]
            bound[j] = y
            print(len(steps),len(y), exp)
            # plt.plot(steps, y,  label=exp, linestyle=ls[j], marker=m[j],color=c[k],alpha=a[j])
            plt.plot(steps, y,  label=exp, linestyle=ls[j], marker=m[j],color=c[j],alpha=a[j])
            j+=1
        k += 1
        # zero_bound = [0]*len(steps)
        # plt.fill_between(steps, bound[0], bound[1],   alpha=0.8, label='parallel', color='w', edgecolor=c[0], hatch='\\') 
        # plt.fill_between(steps, bound[1], zero_bound,  alpha=0.1, label='orthogonal', color=c[1]) #facecolor='C0',
    # 设置xy坐标范围
    # plt.ylim((40, 65))
    plt.xlim((0,250))
    
    # plt.xscale('log')
    
    #xy描述
    plt.ylabel('范数', fontsize=16)
    plt.xlabel("步数", fontsize=16)
    plt.legend(loc='upper right', fontsize=16)

    # title = 'Norm ratio' + param_name
    # plt.title(title, fontsize=9)
    plt.show()
    file_name = 'grad_diff_ch.pdf'
    plt.savefig(file_list_path+file_name, dpi=600)
    plt.close()
    
if __name__ == '__main__':
    parameter_plot()
    segment_plot()
    diff_plot()
    convergence_plot()
