# -*- coding: utf-8 -*-
# @Author : Zhang
# @Email : zl16035056@163.com
# @File : cnn.py


import torch.nn as nn
import warnings

warnings.filterwarnings("ignore")


class Model(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Model, self).__init__()
        self.name = 'CNN'
        self.nn_layer = nn.ModuleList()
        self.layer1 = nn.Sequential(nn.Conv2d(1, 16, kernel_size=8, stride=2, padding=2),
                                    nn.Tanh(),
                                    nn.MaxPool2d(kernel_size=2, stride=1))
        self.nn_layer.append(self.layer1)
        self.layer2 = nn.Sequential(nn.Conv2d(16, 32, kernel_size=4, stride=2, padding=0),
                                    nn.Tanh(),
                                    nn.MaxPool2d(kernel_size=2, stride=1))
        self.nn_layer.append(self.layer2)
        self.fc = nn.Sequential(nn.Linear(4 * 4 * 32, 32),
                                nn.Tanh(),
                                nn.Linear(32, output_dim))
        self.nn_layer.append(self.fc)

    def forward(self, x):
        for id, layer in enumerate(self.nn_layer):
            if id == len(self.nn_layer)-1:
                x = x.view(x.size(0), -1)
            x = layer(x)
        return x
