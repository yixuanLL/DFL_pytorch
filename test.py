# -*- coding: utf-8 -*-
# @Author : yixuan
# @File : dataloader.py


import torch
from torchvision import datasets, transforms

train_dataloader = datasets.MNIST(root='~/data', train=True, download=False, transform=transforms.ToTensor())
x_train = train_dataloader.data.float().unsqueeze(1)#.to(device)
print('finish')