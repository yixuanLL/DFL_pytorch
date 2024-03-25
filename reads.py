import sys
import numpy as np

f = sys.stdin.readline().strip()
# f = 'log_ddp_sgd_MNIST_20240219'
file = open('/local/scratch/yliu270/workspace/DFL_pytorch/logs/standalone/'+f, 'r')
acc = []
lr_list = []
eps_list = []
grad_norm_list = []
grad_perp_norm_list = []
algo_list = []
clip_alpha_list = []
dp_list = []


rounds = 0
l=''
e=''
iid=''
for line in file:
    line = line.strip()
    lr = '#'
    eps = 'non-dp'
    grad_norm = '#'
    grad_perp_norm = '#'
    algo = 'SGD'
    clip_alpha = '#'
    dp = False
    if 'global_round :' in line:
        rounds = int(line.split(':')[1].strip())
    if 'python' in line:
        line = line.split(' ')
        for item in line:
            item = item.strip()
            if 'global_round' in item:
                global_round = int(item.split('=')[1])
            if '--lr' in item:
                lr = item.split('=')[1]
            if '--dp=T' in item:
                dp = True
            if '--DR=T' in item:
                algo = 'DR'
            if '--DRtest=T' in item:
                algo = 'DRtest'
            if '--eps=' in item:
                eps = item.split('=')[1]
            if 'grad_norm' in item:
                grad_norm = item.split('=')[1]
            if 'grad_perp_norm' in item:
                grad_perp_norm = item.split('=')[1]
            if 'clip_paral' in item:
                clip_alpha = item.split('=')[1]

        rounds = global_round  
        lr_list.append(lr)
        eps_list.append(eps)
        grad_norm_list.append(grad_norm)
        grad_perp_norm_list.append(grad_perp_norm)
        algo_list.append(algo)
        clip_alpha_list.append(clip_alpha)
        dp_list.append(dp)
    rounds_str = 'Epoch is: '+str(int(rounds))
    if rounds_str in line:
        acc.append(str(round(float(line.split(' ')[11].split(',')[0])*100, 2)))


# acc = np.array(acc)
# print(acc*100)
# acc_r = acc[0:56].reshape(( len(acc)//4, 4))
# print(acc_r)

i=0

# num_col = 2
print('g_norm:', '\t'.join(set(grad_norm_list)))
print('g_perp_norm:', '\t'.join(set(grad_perp_norm_list)))
print('clip_paral:', '\t'.join(set(clip_alpha_list)))

print('Alg \t clip_paral \t eps \t lr')
while i < len(acc):
    # print(algo_list[i])
    # print(dp_list[i])
    if algo_list[i] == 'SGD' and not dp_list[i]:
        num_col = 1
    elif algo_list[i] == 'SGD' and dp_list[i]:
        algo_list[i] = 'DPSGD'
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
    print(algo_list[i] + '\t'+ clip_alpha_list[i]  + '\t'+eps_list[i]+'\t'+lr_list[i]+'\t'+res)
    i += num_col



# import sys
# import numpy as np

# f = sys.stdin.readline().strip()
# # f = 'log_cpl_20231027'
# file = open('/local/scratch/yliu270/workspace/DFL_pytorch/logs/standalone/'+f, 'r')
# acc = []
# lr = []
# eps=[]
# iid_list = []
# norm = []
# p_norm= []
# rounds = 0
# l=''
# e=''
# iid=''
# for line in file:
#     line = line.strip()
#     if 'global_round :' in line:
#         rounds = int(line.split(':')[1].strip())
#     if 'csv' in line:
#         # if 'CIFAR10' not in line:
#         #     continue
#         print(line)
#         line = line.split('/')
#         l = line[-1].split('-')[0]
#         n = line[-1].split('-')[-2]
#         p = line[-1].split('-')[-3]
#         e = line[-2]
#         iid = line[-4]
#         lr.append(l)
#         eps.append(e)
#         iid_list.append(iid)
#         norm.append(n)
#         p_norm.append(p)
#     rounds_str = 'Epoch is: '+str(int(rounds))
#     if rounds_str in line:
#         acc.append(str(round(float(line.split(' ')[11].split(',')[0])*100, 2)))


# # acc = np.array(acc)
# # print(acc*100)
# # acc_r = acc[0:56].reshape(( len(acc)//4, 4))
# # print(acc_r)

# i=0
# num_col = max(len(set(p_norm)), len(set(norm)))
# # num_col = 1
# while i < len(acc):
#     res = '\t'.join(acc[i:i+num_col])
#     # print(iid_list[i] + '\t' + eps[i]+'\t'+lr[i]+'\t'+res)
#     print(eps[i]+'\t'+lr[i]+'\t'+res)
#     i += num_col