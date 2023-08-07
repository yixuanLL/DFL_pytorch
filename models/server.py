# -*- coding: utf-8 -*-
# @Author : Zhang
# @Email : zl16035056@163.com
# @File : server.py

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn


def add_weights(num_vars, model_state, agg_model_state):
    # for i in range(num_vars):
    #     if not len(agg_model_dict):
    #         a = torch.unsqueeze(model_dict[i], 0)
    #     else:
    #         b = torch.cat([agg_model_dict[i], torch.unsqueeze(model_dict[i], 0)], 0)
    return [torch.unsqueeze(model_state[i], 0)
            if not len(agg_model_state) else torch.cat([agg_model_state[i], torch.unsqueeze(model_state[i], 0)], 0) for i in range(num_vars)]


class FedAvg:
    def __init__(self):
        self.__model_state = []
        self.num_vars = None
        self.shape_vars = None

    def aggregate(self, model_state):
        if not self.shape_vars:
            self.shape_vars = [var.shape for var in model_state]

        self.num_vars = len(model_state)
        update_model_state = [state.flatten() for state in model_state]
        self.__model_state = add_weights(self.num_vars, update_model_state, self.__model_state)

    def average(self):
        mean_updates = [torch.mean(self.__model_state[i], 0).reshape(self.shape_vars[i]) for i in range(self.num_vars)]
        self.__model_state = []
        return mean_updates


class Server:
    def __init__(self, num_clients, model, sample_ratio, x_test, y_test):
        super(Server, self).__init__()
        self.num_clients = num_clients
        self.sample_ratio = sample_ratio

        # self.model = CNN(input_dim=1, output_dim=10)
        # self.model = model(input_dim=1, output_dim=10)
        self.model = model(784,10)
        self.state_dict_key = self.model.state_dict().keys()

        self.num_vars = None
        self.shape_vars = None
        self.__alg = None
        self.__epsilons = None
        self.x_test = x_test
        self.y_test = y_test

    def init_global_model(self):
        return self.model

    def sample_clients(self, candidates):
        m = int(self.num_clients * self.sample_ratio)

        if len(candidates) < m:
            return []
        else:
            participants = list(np.random.permutation(candidates))[0:m]
            return participants

    def init_alg(self, dp=True):
        self.__alg = FedAvg()
        print('\nUsing FedAvg algorithm!!!\n')

    def aggregate(self, model_state):
        self.__alg.aggregate(model_state)

    def update(self):
        mean_state = self.__alg.average()

        mean_updates = dict(zip(self.state_dict_key, mean_state))

        self.model.load_state_dict(mean_updates)
        return self.model

    def test(self, model):
        model.eval().to('cuda')
        data_loader = TensorDataset(self.x_test, self.y_test)
        data_loader = DataLoader(data_loader, batch_size=128, shuffle=True)
        criterion = nn.CrossEntropyLoss()
        test_loss = 0
        test_acc = 0

        with torch.no_grad():
            for x_test, y_test in data_loader:
                x_test, y_test = x_test.to('cuda'), y_test.to('cuda')

                output = model(x_test)
                loss = criterion(output, y_test)

                _, test_pred = torch.max(output, 1)

                correct = (test_pred == y_test).sum()

                test_acc += correct.item()
                test_loss += loss.item()

        print()
        return test_acc / len(self.x_test), test_loss / len(self.y_test)
