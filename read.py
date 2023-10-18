import sys
import numpy as np

f = sys.stdin.readline().strip()
file = open('/home/yliu270/workspace/DFL_pytorch/logs/'+f, 'r')
acc = []
lr = []
eps=[]
iid_list = []
l=''
e=''
iid=''
for line in file:
    line = line.strip()
    if 'csv' in line:
        if 'MNIST' not in line:
            continue
        print(line)
        line = line.split('/')
        l = line[-1].split('-')[0]
        e = line[-2]
        iid = line[-4]
        lr.append(l)
        eps.append(e)
        iid_list.append(iid)
    if 'round 49' in line:
        acc.append(str(round(float(line.split(' ')[7])*100, 2)))


# acc = np.array(acc)
# print(acc*100)
# acc_r = acc[0:56].reshape(( len(acc)//4, 4))
# print(acc_r)

i=0
num_col = 3
while i < len(acc):
    res = '\t'.join(acc[i:i+num_col])
    print(iid_list[i] + '\t' + eps[i]+'\t'+lr[i]+'\t'+res)
    i += num_col