# -*- coding: utf-8 -*-
# @Author : Zhang
# @Email : zl16035056@163.com
# @File : server.py

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import copy
from opt_einsum.contract import contract


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

    def aggregate(self, model_state, global_last_model=None):
        if not self.shape_vars:
            self.shape_vars = [var.shape for var in model_state] # model weight

        self.num_vars = len(model_state) # num of layers
        update_model_state = [state.flatten() for state in model_state] # flatted model params per layer
        self.__model_state = add_weights(self.num_vars, update_model_state, self.__model_state) # concat weights

    def average(self, global_last_model=None):
        mean_updates = [torch.mean(self.__model_state[i].type(torch.float), 0).reshape(self.shape_vars[i]) for i in range(self.num_vars)] # mean weights
        self.__model_state = []
        return mean_updates

class FedDRDP:
    def __init__(self, perp_grad_norm, clip_paral, noise_multiplier, noise_multiplier_2):
        self.__model_grads = []
        self.num_vars = None
        self.shape_vars = None
        self.last_grad = None
        self.perp_grad_norm = perp_grad_norm
        self.clip_paral = clip_paral
        self.noise_multiplier = noise_multiplier
        self.noise_multiplier_2 = noise_multiplier_2
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def aggregate(self, model_state, global_last_model):
        if not self.shape_vars:
            self.shape_vars = [var.shape for var in model_state] # model weight

        self.num_vars = len(model_state) # num of layers
        update_model_grad = [state.flatten()-lg.flatten() for state, lg in zip(model_state, global_last_model)] # flatted model params per layer
        self.__model_grads = add_weights(self.num_vars, update_model_grad, self.__model_grads) # concat weights
        if self.last_grad == None:
            self.last_grad = [g.flatten() for g in global_last_model]
        else:
            self.last_grad = [torch.mean(g.reshape(2,-1), dim=0) for g in self.__model_grads]

    def dr_process(self):
        gi_perp, alpha_i = self.decompose_grad()   
        g_perp = self.clip_g_perp(gi_perp) 
        g_perp_clean = copy.deepcopy(g_perp) 
        self.add_noise_sum(g_perp, self.noise_multiplier, self.perp_grad_norm)

        clip_p = self.clip_paral
        alpha = self.clip(alpha_i, clip_p)
        alpha_clean = copy.deepcopy(alpha)
        self.add_noise_sum(alpha, self.noise_multiplier_2, clip_p) 

        self.recover_grad(g_perp, alpha) 
    
    def decompose_grad(self):
        num_clients = len(self.__model_grads[0])
        last_grad_norms = [g.norm(2, dim=-1) for g in self.last_grad] # norm of per laryer of last gradient
        paral_alpha = [torch.sum(g.reshape(num_clients, -1)*lg, dim=1)/(lg_norm*lg_norm) for (g, lg, lg_norm) in zip(self.__model_grads, self.last_grad, last_grad_norms)]
        gi_paral = [paral.reshape([num_clients]+[1]*len(lg.shape)) * torch.tile(lg.unsqueeze(0),[num_clients]+[1]*len(lg.shape)) for paral, lg in zip(paral_alpha, self.last_grad)]
        gi_perp = [(g-gl) for g, gl in zip(self.__model_grads, gi_paral)] 
        return gi_perp, paral_alpha
    
    def clip_g_perp(self, g_perp):
        per_param_norms = [g.norm(2, dim=-1) for g in g_perp] # norm of per laryer of per sample gradient
        per_sample_norms = torch.stack(per_param_norms, dim=1).norm(2, dim=1) # norm of per sample gradient
        per_sample_clip_factor = (self.perp_grad_norm / (per_sample_norms + 1e-6)).clamp(max=1.0) # clip [ max min ]
        g_perp_clipped = []
        for p in g_perp:
            grad = contract("i,i...", per_sample_clip_factor, p) # mutiply [128] * [128, 16, 1, 8, 8] -> [16, 1, 8, 8] clip & sum
            g_perp_clipped.append(grad)
        return g_perp_clipped

    def clip(self, vec, clip_bound):
        norm = torch.stack(vec, dim=1).norm(2, dim=1)
        clip_factor = (
            clip_bound / (norm + 1e-6)
        ).clamp(max=1.0)
        vec = [torch.sum(clip_factor * v) for v in vec]
        return vec

    def add_noise_sum(self, vec, noise_multiplier, sensitivity):
        """
        Adds noise to clipped gradients. Stores clipped and noised result in ``p.grad``
        """
        std = noise_multiplier * sensitivity
        for v in vec:
            noise = torch.normal(mean=0, std=std, size=v.shape, device=self.device, generator=None)
            v += noise
        return vec

    def recover_grad(self, g_perp_noisy, alpha_noisy):
        g_noisy = [(gp + a * lg)/len(self.__model_grads[0]) for gp, a, lg in zip(g_perp_noisy, alpha_noisy, self.last_grad)]
        self.last_grad = copy.deepcopy(g_noisy) 
        return g_noisy

    def average(self, global_last_model=None):
        self.dr_process()
        mean_updates = [global_last_model[i]+self.last_grad[i].type(torch.float).reshape(self.shape_vars[i]) for i in range(self.num_vars)]
        # mean_updates = [torch.mean(self.__model_state[i].type(torch.float), 0).reshape(self.shape_vars[i]) for i in range(self.num_vars)] # mean weights
        self.__model_grads = []
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
    def __init__(self, num_clients, model, sample_ratio, x_test, y_test, model_param, device, perp_grad_norm, clip_paral, noise_multiplier, noise_multiplier_2):
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
        self.perp_grad_norm = perp_grad_norm
        self.clip_paral = clip_paral
        self.noise_multiplier = noise_multiplier
        self.noise_multiplier_2 = noise_multiplier_2

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
        if FLalg == 'FedDRDP': 
            self.__alg = FedDRDP(self.perp_grad_norm, self.clip_paral, self.noise_multiplier, self.noise_multiplier_2)
            print('\nUsing FedDRDP algorithm!!!\n')    
        else:
            self.__alg = FedAvg()
            print('\nUsing FedAvg algorithm!!!\n')
        

    def aggregate(self, model_state, global_last_model):
        self.__alg.aggregate(model_state, global_last_model)

    def update(self, global_last_model):
        mean_state = self.__alg.average(global_last_model)

        mean_updates = dict(zip(self.state_dict_key, mean_state))

        self.model.load_state_dict(mean_updates)
        return self.model

    def test(self, model):
        model.eval() #.to(self.device)
        data_loader = TensorDataset(self.x_test, self.y_test)
        data_loader = DataLoader(data_loader, batch_size=128, shuffle=True)
        criterion = nn.CrossEntropyLoss()
        test_loss = 0
        test_acc = 0

        with torch.no_grad():
            for x_test, y_test in data_loader:
                # x_test, y_test = x_test.to(self.device), y_test.to(self.device)

                output = model(x_test)
                loss = criterion(output, y_test)

                _, test_pred = torch.max(output, 1)

                correct = (test_pred == y_test).sum()

                test_acc += correct.item()
                test_loss += loss.item()

        # print()
        return test_acc / len(self.x_test), test_loss / len(self.y_test)
