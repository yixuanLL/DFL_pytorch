import sys
import numpy as np

f = sys.stdin.readline().strip()
# f = 'log_ddp_sgd_MNIST_20240219'
file = open('/home/yliu270/workspace/DFL_pytorch/logs/'+f, 'r')
acc = []
lr_list = []
glr_list = []
sample_ratio_list = []
eps_list = []
grad_norm_list = []
grad_perp_norm_list = []
FLalg_list = []
clip_alpha_list = []
lr = '#'
glr = '#'
sample_ratio = '#'
eps = 'non-dp'
grad_norm = '#'
grad_perp_norm = '#'
FLalg = 'Non'
clip_alpha = '#'
# lr = []
# eps=[]
# iid_list = []
# norm = []
# p_norm= []
rounds = 0
l=''
e=''
iid=''
for line in file:
    line = line.strip()
    if 'global_round :' in line:
        rounds = int(line.split(':')[1].strip())/1-1
    # if 'csv' in line:
    #     # if 'CIFAR10' not in line:
    #     #     continue
    #     print(line)
    #     line = line.split('/')
    #     l = line[-1].split('-')[0]
    #     n = line[-1].split('-')[-2]
    #     p = line[-1].split('-')[-3]
    #     e = line[-2]
    #     iid = line[-4]
    #     lr.append(l)
    #     eps.append(e)
    #     iid_list.append(iid)
    #     norm.append(n)
    #     p_norm.append(p)
    if 'python' in line:
        line = line.split(' ')
        for item in line:
            item = item.strip()
            if 'local_round' in item:
                local_round = int(item.split('=')[1])
            if 'global_round' in item:
                global_round = int(item.split('=')[1])
            if '--lr' in item:
                lr = item.split('=')[1]
            if 'glr' in item:
                glr = item.split('=')[1]
            if 'sample_ratio' in item:
                sample_ratio = item.split('=')[1]
            if '--eps=' in item:
                eps = item.split('=')[1]
            if 'grad_norm' in item:
                grad_norm = item.split('=')[1]
            if 'grad_perp_norm' in item:
                grad_perp_norm = item.split('=')[1]
            if 'FLalg' in item:
                FLalg = item.split('=')[1]
            if 'clip_paral' in item:
                clip_alpha = item.split('=')[1]

        rounds = global_round / local_round - 1 
        lr_list.append(lr)
        glr_list.append(glr)
        sample_ratio_list.append(sample_ratio)
        eps_list.append(eps)
        grad_norm_list.append(grad_norm)
        grad_perp_norm_list.append(grad_perp_norm)
        FLalg_list.append(FLalg)
        clip_alpha_list.append(clip_alpha)
    rounds_str = 'round '+str(int(rounds))
    if rounds_str in line:
        acc.append(str(round(float(line.split(' ')[7])*100, 2)))


# acc = np.array(acc)
# print(acc*100)
# acc_r = acc[0:56].reshape(( len(acc)//4, 4))
# print(acc_r)

i=0

# num_col = 2
print('g_norm:', '\t'.join(set(grad_norm_list)))
print('g_perp_norm:', '\t'.join(set(grad_perp_norm_list)))
print('sample_ratio:', '\t'.join(set(sample_ratio_list)))
print('clip_paral:', '\t'.join(set(clip_alpha_list)))

print('FL \t clip_paral \t glr \t eps \t lr')
while i < len(acc):
    if FLalg_list[i] == 'FedAvg':
        num_col = 1
    elif FLalg_list[i] == 'FedDPAvg' or FLalg_list[i] == 'FedDPAdam':
        tmp = set(grad_norm_list)
        try:
            tmp.remove('#')
        except:
            pass
        num_col = len(tmp)
    else:
        tmp = set(grad_perp_norm_list)
        try:
            tmp.remove('#')
        except:
            pass
        num_col = len(tmp)

    res = '\t'.join(acc[i:i+num_col])
    # print(iid_list[i] + '\t' + eps[i]+'\t'+lr[i]+'\t'+res)
    print(FLalg_list[i] + '\t'+ clip_alpha_list[i]  + '\t'+ glr_list[i]+'\t'+eps_list[i]+'\t'+lr_list[i]+'\t'+res)
    i += num_col