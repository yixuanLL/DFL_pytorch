import sys
import numpy as np

f = sys.stdin.readline().strip()
file = open('/home/yixuan/workspace/DFL_pytorch/logs/'+f, 'r')
acc = []
for line in file:
    line = line.strip()
    if 'csv' in line:
        print(line)
    if 'round 49' in line:
        acc.append(float(line.split(' ')[7]))

acc = np.array(acc)
print(acc*100)
# acc_r = acc[0:56].reshape(( len(acc)//4, 4))
# print(acc_r)
