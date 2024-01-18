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
    if 'tensor' in line:
        continue
    print(line)

