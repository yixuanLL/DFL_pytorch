# -*- coding: utf-8 -*-
# @Author : Zhang
# @Email : zl16035056@163.com
# @File : cnn.py


import torch.nn as nn
import warnings
import torch
warnings.filterwarnings("ignore")



class Model(nn.Module):
    # def __init__(self, in_channels=3, input_norm=None, **kwargs):
    def __init__(self, input_dim, output_dim, **kwargs):
        super(Model, self).__init__()
        self.name = 'CNN_5layer'
        # self.in_channels = in_channels
        self.in_channels = 3 #for cifar10
        self.features = None
        self.classifier = None
        self.norm = None
        self.input_norm=None
        self.input_dim=input_dim
        self.output_dim=output_dim
        self.nn_layer = nn.ModuleList()
        self.build(self.input_norm, **kwargs)

    def build(self, input_norm=None, num_groups=None,
              bn_stats=None, size=None):

        if self.in_channels == 3:
            if size == "small":
                cfg = [16, 16, 'M', 32, 32, 'M', 64, 'M']
            else:
                cfg = [32, 32, 'M', 64, 64, 'M', 128, 128, 'M']

            self.norm = nn.Identity()
        else:
            if size == "small":
                cfg = [16, 16, 'M', 32, 32]
            else:
                cfg = [64, 'M', 64]
            if input_norm is None:
                self.norm = nn.Identity()
            elif input_norm == "GroupNorm":
                self.norm = nn.GroupNorm(num_groups, self.in_channels, affine=False)
            else:
                self.norm = lambda x: standardize(x, bn_stats)

        layers = []
        act = nn.Tanh

        c = self.in_channels
        for v in cfg:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = nn.Conv2d(c, v, kernel_size=3, stride=1, padding=1)

                layers += [conv2d, act()]
                c = v

        self.features = nn.Sequential(*layers)
        self.nn_layer.append(self.features)

        if self.in_channels == 3:
            hidden = 128
            self.classifier = nn.Sequential(nn.Linear(c * 4 * 4, hidden), act(), nn.Linear(hidden, self.output_dim))
        else:
            self.classifier = nn.Linear(c * 4 * 4, self.output_dim)
        self.nn_layer.append(self.classifier)

    def forward(self, x):
        if self.in_channels != 3:
            x = self.norm(x.view(-1, self.in_channels, 8, 8))
        for id, layer in enumerate(self.nn_layer):
            if id == len(self.nn_layer)-1:
                x = x.view(x.size(0), -1)
            x = layer(x)
        return x

def standardize(x, bn_stats):
    if bn_stats is None:
        return x

    bn_mean, bn_var = bn_stats

    view = [1] * len(x.shape)
    view[1] = -1
    x = (x - bn_mean.view(view)) / torch.sqrt(bn_var.view(view) + 1e-5)

    # if variance is too low, just ignore
    x *= (bn_var.view(view) != 0).float()
    return x