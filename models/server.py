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
import math


def add_weights(num_vars, model_state, agg_model_state):
    # for i in range(num_vars):
    #     if not len(agg_model_dict):
    #         a = torch.unsqueeze(model_dict[i], 0)
    #     else:
    #         b = torch.cat([agg_model_dict[i], torch.unsqueeze(model_dict[i], 0)], 0)
    return [torch.unsqueeze(model_state[i], 0)
            if not len(agg_model_state) else torch.cat([agg_model_state[i], torch.unsqueeze(model_state[i], 0)], 0) for i in range(num_vars)]

def clip(grads, clip_C):
    per_param_norms = [g.norm(2, dim=-1) for g in grads] # norm of per laryer of per sample gradient
    per_sample_norms = torch.stack(per_param_norms, dim=1).norm(2, dim=1) # norm of per sample gradient
    # print(per_sample_norms[0:2])
    per_sample_clip_factor = (clip_C / (per_sample_norms + 1e-6)).clamp(max=1.0) # clip [ max min ]
    grads_clipped = []
    for p in grads:
        # grad = contract("i,i...", per_sample_clip_factor, p) # mutiply [128] * [128, 16, 1, 8, 8] -> [16, 1, 8, 8] clip & sum
        grad = per_sample_clip_factor.reshape(len(p),1) * p # mutiply [128] * [128, 16, 1, 8, 8] -> [128, 16, 1, 8, 8] clip & sum
        grads_clipped.append(grad)
    return grads_clipped

def add_noise_ddp(vec, noise_multiplier, sensitivity, num_clients, device):
    std = noise_multiplier * sensitivity / math.sqrt(num_clients)
    for v in vec:
        noise = torch.normal(mean=0, std=std, size=v.shape, device=device, generator=None)
        v += noise
    return vec

def grad_flat(param, device):
    vec = torch.tensor([]).to(device)
    for p in param:
        vec = torch.cat((vec, p.reshape(-1).to(device)))
    return vec

class FedAvg:
    def __init__(self, glr):
        self.__model_state = []
        self.__model_grads = []
        self.num_vars = None
        self.shape_vars = None
        self.glr = glr

    # upload model updates
    def aggregate(self, model_state, global_last_model=None):
        if not self.shape_vars:
            self.shape_vars = [var.shape for var in model_state] # model weight
        self.num_vars = len(model_state) # num of layers
        update_model_grad = [state.flatten()-lg.flatten() for state, lg in zip(model_state, global_last_model)] # flatted model params per layer
        self.__model_grads = add_weights(self.num_vars, update_model_grad, self.__model_grads) # concat weights

    def average(self, global_last_model=None):
        # for g in self.__model_grads:
        #     print (g)
        g_mean = [torch.mean(torch.tensor(g, dtype=float), dim=0) for g in self.__model_grads]
        mean_updates = [global_last_model[i]+(self.glr * g_mean[i].type(torch.float)).reshape(self.shape_vars[i]) for i in range(self.num_vars)] # mean weights
        self.__model_grads = []
        return mean_updates        

    # upload model parameters
    # def aggregate(self, model_state, global_last_model=None):
    #     if not self.shape_vars:
    #         self.shape_vars = [var.shape for var in model_state] # model weight

    #     self.num_vars = len(model_state) # num of layers
    #     update_model_state = [state.flatten() for state in model_state] # flatted model params per layer
    #     self.__model_state = add_weights(self.num_vars, update_model_state, self.__model_state) # concat weights

    # def average(self, global_last_model=None):
    #     mean_updates = [torch.mean(self.__model_state[i].type(torch.float), 0).reshape(self.shape_vars[i]) for i in range(self.num_vars)] # mean weights
    #     self.__model_state = []
    #     return mean_updates

    
class FedDPAvg:
    def __init__(self, grad_norm, budget_accountant, glr):
        self.__model_grads = []
        self.num_vars = None
        self.shape_vars = None
        self.grad_norm = grad_norm
        self.noise_multiplier = budget_accountant.noise_multiplier
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.glr = glr

    def aggregate(self, model_state, global_last_model=None):
        if not self.shape_vars:
            self.shape_vars = [var.shape for var in model_state] # model weight
        self.num_vars = len(model_state) # num of layers
        update_model_grad = [state.flatten()-lg.flatten() for state, lg in zip(model_state, global_last_model)] # flatted model params per layer
        self.__model_grads = add_weights(self.num_vars, update_model_grad, self.__model_grads) # concat weights

    def average(self, global_last_model=None):
        num_clients = len(self.__model_grads[0])
        gi = clip(self.__model_grads, self.grad_norm)
        add_noise_ddp(gi, self.noise_multiplier, self.grad_norm, num_clients, self.device)
        g_mean = [torch.mean(g, dim=0) for g in gi]
        mean_updates = [global_last_model[i]+(self.glr * g_mean[i].type(torch.float)).reshape(self.shape_vars[i]) for i in range(self.num_vars)] # mean weights
        self.__model_grads = []
        return mean_updates
    

class FedDPAdam:
    def __init__(self, grad_norm, budget_accountant, glr):
        self.__model_grads = []
        self.num_vars = None
        self.shape_vars = None
        self.grad_norm = grad_norm
        self.noise_multiplier = budget_accountant.noise_multiplier
        self.m = None
        self.v = None
        self.glr = glr
        self.opt = 'momentum'
        print('opt:', self.opt)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def aggregate(self, model_state, global_last_model=None):
        if not self.shape_vars:
            self.shape_vars = [var.shape for var in model_state] # model weight
        self.num_vars = len(model_state) # num of layers
        update_model_grad = [state.flatten()-lg.flatten() for state, lg in zip(model_state, global_last_model)] # flatted model params per layer
        self.__model_grads = add_weights(self.num_vars, update_model_grad, self.__model_grads) # concat weights

    def average(self, global_last_model=None):
        num_clients = len(self.__model_grads[0])
        gi = self.clip(self.__model_grads, self.grad_norm)
        self.add_noise_ddp(gi, self.noise_multiplier, self.grad_norm, num_clients, self.device)
        g = [torch.mean(g, dim=0) for g in gi]

        if self.opt == 'adam':
            beta1 = 0.5
            beta2 = 0.99
            tau = 10e-3
            if self.m == None:
                self.m = [(1-beta1) * gg/num_clients for gg in g]
                self.v = [(1-beta2) * (gg/num_clients)**2 for gg in g]
            else:
                self.m = [beta1 * m + (1-beta1) * gg/num_clients for m, gg in zip(self.m, g)]
                self.v = [beta2 * v + (1-beta2) * (gg/num_clients)**2 for v, gg in zip(self.v, g)]
            mean_updates = [global_last_model[i]+(self.glr * self.m[i]/(torch.sqrt(self.v[i]) + tau)).reshape(self.shape_vars[i]) for i in range(self.num_vars)] # mean weights
        
        if self.opt == 'momentum':
            beta = 0.9
            if self.v == None:
                self.v = g
            else:
                self.v = [beta * v + gg for v, gg in zip(self.v, g)]
            mean_updates = [global_last_model[i] + (self.glr * self.v[i]).reshape(self.shape_vars[i]) for i in range(self.num_vars)] # mean weights

        self.__model_grads = []
        return mean_updates

    def clip(self, grads, clip_C):
        per_param_norms = [g.norm(2, dim=-1) for g in grads] # norm of per laryer of per sample gradient
        per_sample_norms = torch.stack(per_param_norms, dim=1).norm(2, dim=1) # norm of per sample gradient
        per_sample_clip_factor = (clip_C / (per_sample_norms + 1e-6)).clamp(max=1.0) # clip [ max min ]
        grads_clipped = []
        for p in grads:
            # grad = contract("i,i...", per_sample_clip_factor, p) # mutiply [128] * [128, 16, 1, 8, 8] -> [16, 1, 8, 8] clip & sum
            grad = per_sample_clip_factor.reshape(len(p),1) * p # mutiply [128] * [128, 16, 1, 8, 8] -> [128, 16, 1, 8, 8] clip & sum
            grads_clipped.append(grad)
        return grads_clipped

    def add_noise_ddp(self, vec, noise_multiplier, sensitivity, num_clients, device):
        std = noise_multiplier * sensitivity / math.sqrt(num_clients)
        for v in vec:
            noise = torch.normal(mean=0, std=std, size=v.shape, device=device, generator=None)
            v += noise
        return vec

class FedDPDIFF:
    def __init__(self, perp_grad_norm, budget_accountant, glr):
        self.__model_grads = []
        self.num_vars = None
        self.shape_vars = None
        self.last_grad = None
        self.perp_grad_norm = perp_grad_norm
        self.noise_multiplier = budget_accountant.noise_multiplier
        self.glr = glr
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def aggregate(self, model_state, global_last_model):
        if not self.shape_vars:
            self.shape_vars = [var.shape for var in model_state] # model weight

        self.num_vars = len(model_state) # num of layers
        update_model_grad = [state.flatten()-lg.flatten() for state, lg in zip(model_state, global_last_model)] # flatted model params per layer
        self.__model_grads = add_weights(self.num_vars, update_model_grad, self.__model_grads) # concat weights
        if self.last_grad == None:
            self.last_grad = [g.flatten() for g in global_last_model]
        # else:
        #     self.last_grad = [torch.mean(g.reshape(len(g),-1), dim=0) for g in self.__model_grads]
    
    def cpl_process(self):
        num_clients = len(self.__model_grads[0])
        #difference
        gi_delta = [g.reshape(num_clients, -1)-lg for g, lg in zip(self.__model_grads, self.last_grad)]
        gi_delta = clip(gi_delta, self.perp_grad_norm)
        add_noise_ddp(gi_delta, self.noise_multiplier, self.perp_grad_norm, num_clients, self.device)
        g_delta = [torch.mean(g, dim=0) for g in gi_delta]
        #recover
        # g_noisy = [(gd + lg)/num_clients for gd, lg in zip(g_delta, self.last_grad)]
        g_noisy = [gd + lg for gd, lg in zip(g_delta, self.last_grad)]
        self.last_grad = copy.deepcopy(g_noisy) 
        

    def average(self, global_last_model=None):
        self.cpl_process()
        mean_updates = [global_last_model[i] + self.glr * self.last_grad[i].type(torch.float).reshape(self.shape_vars[i]) for i in range(self.num_vars)]
        self.__model_grads = []
        return mean_updates

class FedDPDIFF3:
    def __init__(self, perp_grad_norm, alpha, budget_accountant, glr):
        self.__model_grads = []
        self.num_vars = None
        self.shape_vars = None
        self.last_grad = None
        self.perp_grad_norm = perp_grad_norm
        self.noise_multiplier = budget_accountant.noise_multiplier
        self.glr = glr
        self.alpha = alpha
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def aggregate(self, model_state, global_last_model):
        if not self.shape_vars:
            self.shape_vars = [var.shape for var in model_state] # model weight

        self.num_vars = len(model_state) # num of layers
        update_model_grad = [state.flatten()-lg.flatten() for state, lg in zip(model_state, global_last_model)] # flatted model params per layer
        self.__model_grads = add_weights(self.num_vars, update_model_grad, self.__model_grads) # concat weights
        if self.last_grad == None:
            self.last_grad = [g.flatten() for g in global_last_model]
        # else:
        #     self.last_grad = [torch.mean(g.reshape(len(g),-1), dim=0) for g in self.__model_grads]
    
    def cpl_process(self):
        num_clients = len(self.__model_grads[0])
        #difference
        
        gi_delta = [g.reshape(num_clients, -1)-self.alpha*lg for g, lg in zip(self.__model_grads, self.last_grad)]
        # g_delta = self.clip_g_perp(gi_delta) 
        # self.add_noise_sum(g_delta, self.noise_multiplier, self.perp_grad_norm)
        gi_delta = clip(gi_delta, self.perp_grad_norm)
        add_noise_ddp(gi_delta, self.noise_multiplier, self.perp_grad_norm, num_clients, self.device)
        g_delta = [torch.mean(g, dim=0) for g in gi_delta]
        #recover
        # g_noisy = [(gd + lg)/num_clients for gd, lg in zip(g_delta, self.last_grad)]
        g_noisy = [gd + self.alpha*lg for gd, lg in zip(g_delta, self.last_grad)]
        self.last_grad = copy.deepcopy(g_noisy) 
        

    def average(self, global_last_model=None):
        self.cpl_process()
        mean_updates = [global_last_model[i] + self.glr * self.last_grad[i].type(torch.float).reshape(self.shape_vars[i]) for i in range(self.num_vars)]
        self.__model_grads = []
        return mean_updates


class FedDRDP:
    def __init__(self, grad_norm, perp_grad_norm, clip_paral, budget_accountant, glr):
        self.__model_grads = []
        self.num_vars = None
        self.shape_vars = None
        self.last_grad = None
        self.perp_grad_norm = perp_grad_norm
        self.grad_norm = grad_norm
        self.clip_paral = clip_paral
        self.m = 0
        self.v = 0
        self.steps = 0
        self.noise_multiplier = budget_accountant.noise_multiplier
        self.noise_multiplier_2 = budget_accountant.noise_multiplier_2
        self.noise_multiplier_3 = budget_accountant.noise_multiplier_3
        self.glr = glr
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def aggregate(self, model_state, global_last_model):
        if not self.shape_vars:
            self.shape_vars = [var.shape for var in model_state] # model weight

        self.num_vars = len(model_state) # num of layers
        update_model_grad = [state.flatten()-lg.flatten() for state, lg in zip(model_state, global_last_model)] # flatted model params per layer
        self.__model_grads = add_weights(self.num_vars, update_model_grad, self.__model_grads) # concat weights
        if self.last_grad == None:
            self.last_grad = [g.flatten() for g in global_last_model]
        # else:
            # self.last_grad = [torch.mean(g.reshape(len(g),-1), dim=0) for g in self.__model_grads]

    def dr_process(self):
        gi_perp, alpha_i = self.decompose_grad()   
        gi_perp = clip(gi_perp, self.perp_grad_norm)
        add_noise_ddp(gi_perp, self.noise_multiplier, self.perp_grad_norm, len(self.__model_grads[0]), self.device)
        g_perp = [torch.mean(g, dim=0) for g in gi_perp]

        clip_p = self.clip_paral
        alpha = self.clip_alpha(alpha_i, clip_p)
        add_noise_ddp(alpha, self.noise_multiplier_2, clip_p, len(self.__model_grads[0]), self.device)
        alpha = [torch.mean(a) for a in alpha]

        self.recover_grad(g_perp, alpha) 
    
    def decompose_grad(self):
        num_clients = len(self.__model_grads[0])
        last_grad_norms = [g.norm(2, dim=-1) for g in self.last_grad] # norm of per laryer of last gradient
        paral_alpha = [torch.sum(g.reshape(num_clients, -1)*lg, dim=1)/(lg_norm*lg_norm) for (g, lg, lg_norm) in zip(self.__model_grads, self.last_grad, last_grad_norms)]
        gi_paral = [paral.reshape([num_clients]+[1]*len(lg.shape)) * torch.tile(lg.unsqueeze(0),[num_clients]+[1]*len(lg.shape)) for paral, lg in zip(paral_alpha, self.last_grad)]
        gi_perp = [(g-gl) for g, gl in zip(self.__model_grads, gi_paral)] 
        return gi_perp, paral_alpha

    def clip_alpha(self, vec, clip_bound):
        norm = torch.stack(vec, dim=1).norm(2, dim=1)
        # print(norm[0:2])
        clip_factor = (
            clip_bound / (norm + 1e-6)
        ).clamp(max=1.0)
        # vec = [torch.sum(clip_factor * v) for v in vec]
        vec = [clip_factor * v for v in vec]
        return vec

    def recover_grad(self, g_perp_noisy, alpha_noisy):
        g_noisy = [gp + a * lg for gp, a, lg in zip(g_perp_noisy, alpha_noisy, self.last_grad)]
        self.last_grad = copy.deepcopy(g_noisy) 
        return g_noisy

    def average(self, global_last_model=None):
        # if self.steps % 30 < 10 and self.steps<800:
        if self.steps!=1:
            self.dr_process()
        else:
            num_clients = len(self.__model_grads[0])
            gi = clip(self.__model_grads, self.grad_norm)
            add_noise_ddp(gi, self.noise_multiplier_3, self.grad_norm, num_clients, self.device)
            g_noisy = [torch.mean(g, dim=0) for g in gi]
            self.last_grad = copy.deepcopy(g_noisy) 
        mean_updates = [global_last_model[i] + self.glr * self.last_grad[i].type(torch.float).reshape(self.shape_vars[i]) for i in range(self.num_vars)]
        self.__model_grads = []
        self.steps += 1
        return mean_updates

class FedDRDPV5:
    def __init__(self, grad_norm, perp_grad_norm, clip_paral, budget_accountant, glr):
        self.__model_grads = []
        self.num_vars = None
        self.shape_vars = None
        self.last_grad = None
        self.perp_grad_norm = perp_grad_norm
        self.grad_norm = grad_norm
        self.clip_paral = clip_paral
        self.m = 0
        self.v = 0
        self.steps = 0
        self.noise_multiplier = budget_accountant.noise_multiplier
        self.noise_multiplier_2 = budget_accountant.noise_multiplier_2
        self.noise_multiplier_3 = budget_accountant.noise_multiplier_3
        self.glr = glr
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def aggregate(self, model_state, global_last_model):
        if not self.shape_vars:
            self.shape_vars = [var.shape for var in model_state] # model weight

        self.num_vars = len(model_state) # num of layers
        update_model_grad = [state.flatten()-lg.flatten() for state, lg in zip(model_state, global_last_model)] # flatted model params per layer
        self.__model_grads = add_weights(self.num_vars, update_model_grad, self.__model_grads) # concat weights
        if self.last_grad == None:
            self.last_grad = [g.flatten() for g in global_last_model]
        # else:
            # self.last_grad = [torch.mean(g.reshape(len(g),-1), dim=0) for g in self.__model_grads]

    def dr_process(self):
        gi_perp, alpha_i = self.decompose_grad()   
        gi_perp = clip(gi_perp, self.perp_grad_norm)
        add_noise_ddp(gi_perp, self.noise_multiplier, self.perp_grad_norm, len(self.__model_grads[0]), self.device)
        g_perp = [torch.mean(g, dim=0) for g in gi_perp]

        clip_p = self.clip_paral
        alpha = self.clip_alpha(alpha_i, clip_p)
        add_noise_ddp(alpha, self.noise_multiplier_2, clip_p, len(self.__model_grads[0]), self.device)
        alpha = [torch.mean(a) for a in alpha]

        self.recover_grad(g_perp, alpha) 
    
    def decompose_grad(self):
        num_clients = len(self.__model_grads[0])
        last_grad_norms = [g.norm(2, dim=-1) for g in self.last_grad] # norm of per laryer of last gradient
        paral_alpha = [torch.sum(g.reshape(num_clients, -1)*lg, dim=1)/(lg_norm*lg_norm) for (g, lg, lg_norm) in zip(self.__model_grads, self.last_grad, last_grad_norms)]
        gi_paral = [paral.reshape([num_clients]+[1]*len(lg.shape)) * torch.tile(lg.unsqueeze(0),[num_clients]+[1]*len(lg.shape)) for paral, lg in zip(paral_alpha, self.last_grad)]
        gi_perp = [(g-gl) for g, gl in zip(self.__model_grads, gi_paral)] 
        return gi_perp, paral_alpha

    def clip_alpha(self, vec, clip_bound):
        norm = torch.stack(vec, dim=1).norm(2, dim=1)
        # print(norm[0:2])
        clip_factor = (
            clip_bound / (norm + 1e-6)
        ).clamp(max=1.0)
        # vec = [torch.sum(clip_factor * v) for v in vec]
        vec = [clip_factor * v for v in vec]
        return vec

    def recover_grad(self, g_perp_noisy, alpha_noisy):
        g_noisy = [gp + a * lg for gp, a, lg in zip(g_perp_noisy, alpha_noisy, self.last_grad)]
        noisy_mean_g = copy.deepcopy(g_noisy) 
        s = 30
        if self.steps % s == 0:
        # if self.steps == 1: # accumulation
            self.last_grad = noisy_mean_g
        else:
            self.last_grad = [(g+lg*(self.steps%s))/(self.steps%s+1) for g, lg in zip(noisy_mean_g, self.last_grad)]
        # normalize for convergence
        last_norm = [p.reshape(-1).norm(2, dim=-1) for p in self.last_grad]
        norm = torch.stack(last_norm).norm(2)
        self.last_grad = [p/norm for p in self.last_grad]
        return g_noisy

    def average(self, global_last_model=None):
        s = 30
        if self.steps % s < 10 and self.steps<80:
        # if self.steps!=1:
            self.dr_process()
        else:
            num_clients = len(self.__model_grads[0])
            gi = clip(self.__model_grads, self.grad_norm)
            add_noise_ddp(gi, self.noise_multiplier_3, self.grad_norm, num_clients, self.device)
            g_noisy = [torch.mean(g, dim=0) for g in gi]
            self.last_grad = copy.deepcopy(g_noisy) 
        mean_updates = [global_last_model[i] + self.glr * self.last_grad[i].type(torch.float).reshape(self.shape_vars[i]) for i in range(self.num_vars)]
        self.__model_grads = []
        self.steps += 1
        return mean_updates


class Server:
    def __init__(self, num_clients, model, sample_ratio, x_test, y_test, model_param, device, grad_norm, perp_grad_norm, clip_paral, budget_accountant, glr):
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
        self.grad_norm = grad_norm
        self.perp_grad_norm = perp_grad_norm
        self.clip_paral = clip_paral
        self.budget_accountant = budget_accountant
        self.glr = glr

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
            self.__alg = FedDRDP(self.grad_norm, self.perp_grad_norm, self.clip_paral, self.budget_accountant, self.glr)
            print('\nUsing FedDRDP algorithm!!!\n')    

        elif FLalg == 'FedDPDIFF': 
            self.__alg = FedDPDIFF(self.perp_grad_norm, self.budget_accountant, self.glr)
            print('\nUsing FedDPDIFF algorithm!!!\n') 
        elif FLalg == 'FedDPDIFF3': 
            self.__alg = FedDPDIFF3(self.perp_grad_norm, self.clip_paral, self.budget_accountant, self.glr)
            print('\nUsing FedDPDIFF3 algorithm!!!\n') 

        elif FLalg == 'FedDPAvg': 
            self.__alg = FedDPAvg(self.grad_norm, self.budget_accountant, self.glr)
            print('\nUsing FedDPAvg algorithm!!!\n')  



        elif FLalg == 'FedDPAdam': 
            self.__alg = FedDPAdam(self.grad_norm, self.budget_accountant, self.glr)
            print('\nUsing FedDPAdam algorithm!!!\n')  

        else:
            self.__alg = FedAvg(self.glr)
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
