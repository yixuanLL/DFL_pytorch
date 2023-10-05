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
            self.shape_vars = [var.shape for var in model_state] # model weight

        self.num_vars = len(model_state) # num of layers
        update_model_state = [state.flatten() for state in model_state] # flatted model params per layer
        self.__model_state = add_weights(self.num_vars, update_model_state, self.__model_state) # concat weights

    def average(self, global_model=None, global_last_grad=None):
        a = self.__model_state
        mean_updates = [torch.mean(self.__model_state[i].type(torch.float), 0).reshape(self.shape_vars[i]) for i in range(self.num_vars)] # mean weights
        self.__model_state = []
        return mean_updates

class FedDrAvg():
    def __init__(self):
        self.__model_state = []
        self.num_vars = None
        self.shape_vars = None
        self.__costheta = []

    def aggregate(self, model_states):
        model_state = model_states[0]
        cos = model_state[1]
        if not self.shape_vars:
            self.shape_vars = [var.shape for var in model_state] # g_perp shape
            self.costheta = [var.shape for var in cos] # cos_shape
        self.num_vars = len(model_state)
        update_model_state = [state.flatten() for state in model_state]
        self.__model_state = add_weights(self.num_vars, update_model_state, self.__model_state)
        self.__costheta = add_weights(self.num_vars, cos, self.__costheta)

    def average(self, global_model, global_last_grad):
        #recover: g_perp+g_paral
        mean_g_perp = [torch.mean(self.__model_state[i], 0).reshape(self.shape_vars[i]) for i in range(self.num_vars)]
        mean_g_paral = [torch.mean(self.__costheta[i]) * global_last_grad[i] for i in range(self.num_vars)]
        global_model = [p.data.to(self.device) for p in global_model.parameters()]
        mean_updates = [-(mean_g_perp[i] + mean_g_paral[i])*0.01 + global_model[i] for i in range(self.num_vars)]
        # temprory *  learning rate!
        # SGD by hand
        self.__model_state = []
        return mean_updates


class Server:
    def __init__(self, num_clients, model, sample_ratio, x_test, y_test, model_param, device):
        super(Server, self).__init__()
        self.num_clients = num_clients
        self.sample_ratio = sample_ratio
        self.model = model(model_param[0], model_param[1])
        self.state_dict_key = self.model.state_dict().keys()

        self.num_vars = None
        self.shape_vars = None
        self.__alg = None
        self.__epsilons = None
        self.x_test = x_test
        self.y_test = y_test
        self.global_last_grad = []
        self.device = device

    def init_global_model(self):
        return self.model

    def sample_clients(self, candidates):
        m = int(self.num_clients * self.sample_ratio)

        if len(candidates) < m:
            return []
        else:
            participants = list(np.random.permutation(candidates))[0:m]
            return participants

    def init_alg(self, dp=True, FLalg='FedAvg'):
        # if FLalg == 'FedAvg':
        if 1==1:
            self.__alg = FedAvg()
            print('\nUsing FedAvg algorithm!!!\n')
        # if FLalg == 'FedDrAvg': ##TODO 跑出来结果不对
        #     self.__alg = FedDrAvg()
        #     print('\nUsing FedDrAvg algorithm!!!\n')            

    def aggregate(self, model_state):
        self.__alg.aggregate(model_state)

    def update(self):
        mean_state = self.__alg.average(self.model, self.global_last_grad)

        mean_updates = dict(zip(self.state_dict_key, mean_state))

        self.model.load_state_dict(mean_updates)
        return self.model

    def test(self, model):
        model.eval().to(self.device)
        data_loader = TensorDataset(self.x_test, self.y_test)
        data_loader = DataLoader(data_loader, batch_size=128, shuffle=True)
        criterion = nn.CrossEntropyLoss()
        test_loss = 0
        test_acc = 0

        with torch.no_grad():
            for x_test, y_test in data_loader:
                x_test, y_test = x_test.to(self.device), y_test.to(self.device)

                output = model(x_test)
                loss = criterion(output, y_test)

                _, test_pred = torch.max(output, 1)

                correct = (test_pred == y_test).sum()

                test_acc += correct.item()
                test_loss += loss.item()

        print()
        return test_acc / len(self.x_test), test_loss / len(self.y_test)
