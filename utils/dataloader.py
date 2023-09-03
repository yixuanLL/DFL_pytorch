# -*- coding: utf-8 -*-
# @Author : yixuan
# @Email : zl16035056@163.com
# @File : dataloader.py


import torch
from torchvision import datasets, transforms
import numpy as np

def loader(name):

    if name == 'MNIST':
        train_dataloader = datasets.MNIST(root='~/data', train=True, download=False, transform=transforms.ToTensor())
        x_train = train_dataloader.data.float().unsqueeze(1)
        y_train = train_dataloader.targets

        indices_train = torch.argsort(y_train)
        sorted_x_train = x_train[indices_train]
        sorted_y_train = y_train[indices_train]

        test_dataloader = datasets.MNIST(root='~/data', train=False, download=False, transform=transforms.ToTensor())
        x_test = test_dataloader.data.float().unsqueeze(1)
        y_test = test_dataloader.targets

    if name == 'CIFAR10':
        transform = transforms.Compose(
            [transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        train_dataloader = datasets.CIFAR10(root='~/data', train=True, download=False, transform=transform)
        train_data = torch.utils.data.DataLoader(train_dataloader, batch_size=50000, shuffle=False, num_workers=0)
        x_train, y_train  = next(iter(train_data))
        # x_train = torch.tensor(train_dataloader.data)
        # y_train = torch.tensor(train_dataloader.targets)
        
        indices_train = torch.argsort(y_train)
        # indices_train = [i for (v, i) in sorted((v, i) for (i, v) in enumerate(y_train))]
        sorted_x_train = x_train[indices_train]
        sorted_y_train = y_train[indices_train]

        test_dataloader = datasets.CIFAR10(root='~/data', train=False, download=False, transform=transform)  
        train_data = torch.utils.data.DataLoader(test_dataloader, batch_size=10000, shuffle=False, num_workers=0)
        x_test, y_test  = next(iter(train_data))      

    print('Using {} dataset!\n'.format(name))


    return sorted_x_train, sorted_y_train, x_test, y_test