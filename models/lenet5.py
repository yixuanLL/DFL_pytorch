# -*- coding: utf-8 -*-
# @Author : Zhang
# @Email : zl16035056@163.com
# @File : cnn.py


import torch.nn as nn
import warnings

warnings.filterwarnings("ignore")


#MNIST
class Model(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Model, self).__init__()
        self.name = 'LeNet5'
        self.layer1 = nn.Sequential(nn.Conv2d(3, 6, kernel_size=5, stride=1, padding=0), # 1, 6 for MNIST
                                    nn.Tanh(),
                                    nn.MaxPool2d(kernel_size=2, stride=2, padding=0))

        self.layer2 = nn.Sequential(nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0),
                                    nn.Tanh(),
                                    nn.MaxPool2d(kernel_size=2, stride=2, padding=0))

        self.fc = nn.Sequential(nn.Linear(16 * 5 * 5, 120), # 16*4*4 for MNIST
                                nn.Tanh(),
                                nn.Linear(120, 84),
                                nn.Tanh(),
                                nn.Linear(84, output_dim))

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x




#CIFAR10
class Model_CIFAR10(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Model_CIFAR10, self).__init__()
        self.name = 'LeNet5'
        self.layer1 = nn.Sequential(nn.Conv2d(3, 6, kernel_size=5, stride=1, padding=0),
                                    nn.MaxPool2d(kernel_size=2, stride=2, padding=0))

        self.layer2 = nn.Sequential(nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0),
                                    nn.MaxPool2d(kernel_size=2, stride=2, padding=0))

        self.fc = nn.Sequential(nn.Linear(16 * 5 * 5, 120),
                                nn.ReLU(),
                                nn.Linear(120, 84),
                                nn.ReLU(),
                                nn.Linear(84, output_dim))

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x