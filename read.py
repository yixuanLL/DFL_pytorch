import sys
import numpy as np

f = sys.stdin.readline().strip()
# f = 'log_cpl_20231027'
file = open('/home/yliu270/workspace/DFL_pytorch/logs/'+f, 'r')
acc = []
lr = []
eps=[]
iid_list = []
norm = []
p_norm= []
rounds = 0
l=''
e=''
iid=''
for line in file:
    line = line.strip()
    if 'global_round :' in line:
        rounds = int(line.split(':')[1].strip())/2-1
    if 'csv' in line:
        # if 'FLamby' not in line:
        #     continue
        print(line)
        line = line.split('/')
        l = line[-1].split('-')[0]
        n = line[-1].split('-')[-2]
        p = line[-1].split('-')[-3]
        e = line[-2]
        iid = line[-4]
        lr.append(l)
        eps.append(e)
        iid_list.append(iid)
        norm.append(n)
        p_norm.append(p)
    rounds_str = 'round '+str(int(rounds))
    if rounds_str in line:
        acc.append(str(round(float(line.split(' ')[7])*100, 2)))


# acc = np.array(acc)
# print(acc*100)
# acc_r = acc[0:56].reshape(( len(acc)//4, 4))
# print(acc_r)

i=0
num_col = max(len(set(p_norm)), len(set(norm)))
while i < len(acc):
    res = '\t'.join(acc[i:i+num_col])
    # print(iid_list[i] + '\t' + eps[i]+'\t'+lr[i]+'\t'+res)
    print(eps[i]+'\t'+lr[i]+'\t'+res)
    i += num_col