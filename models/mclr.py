# -*- coding: utf-8 -*-

import torch.nn as nn
import warnings

warnings.filterwarnings("ignore")

class Model(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Model, self).__init__()
        self.name = 'MCLR'
        self.nn_layer = nn.ModuleList()
        self.linear = nn.Linear(input_dim, output_dim)
        self.nn_layer.append(self.linear)
        self.sigmoid = nn.Sigmoid()
        self.nn_layer.append(self.sigmoid)
        
    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = nn.functional.normalize(x)
        for layer in self.nn_layer:
            x = layer(x)
        return x