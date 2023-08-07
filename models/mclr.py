# -*- coding: utf-8 -*-

import torch.nn as nn
import warnings

warnings.filterwarnings("ignore")

class Model(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Model, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = nn.functional.normalize(x)
        x = self.linear(x)
        outputs = self.sigmoid(x)
        return outputs