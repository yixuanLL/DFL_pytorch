# -*- coding: utf-8 -*-

from torchvision import models
import torch.nn as nn
import torch
from torch.autograd import Variable

class Model(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Model, self).__init__()
        self.name = 'ResNet18_pre'
        self.net = models.resnet18(pretrained=True)

        self.net.classifier = nn.Sequential(
            nn.Linear(in_features=9216, out_features=4096, bias=True),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(in_features=4096, out_features=4096, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=4096, out_features=output_dim, bias=True)
        )

    def forward(self, x:torch.Tensor):
        x = self.net(x)
        return x



