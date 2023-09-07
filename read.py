import sys
import numpy as np

file = open('/home/yixuan/workspace/DFL_pytorch/logs/log_nid_20230905', 'r')
acc = []
for line in file:
    line = line.strip()
    if 'csv' in line:
        print(line)
    if 'round 99' in line:
        acc.append(float(line.split(' ')[7]))

acc = np.array(acc)
acc_r = acc[0:56].reshape(( len(acc)//4, 4))
print(acc_r)
