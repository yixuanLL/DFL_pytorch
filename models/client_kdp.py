# -*- coding: utf-8 -*-
# @Author : Zhang
# @Email : zl16035056@163.com
# @File : client.py


import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from opacus import PrivacyEngine
from functools import reduce
from operator import mul
from utils.kalman_filter import KalmanFilter
import copy

class Client(nn.Module):
    def __init__(self, x_train, y_train, dataset, batch_size, FLalg, dp, DR, DRV2, DRtest,Topk, cpl, kfilter, rate_dr, local_round, grad_norm, grad_perp_norm, lr, momentum, budget_accountant, device, clip_paral):
        super(Client, self).__init__()
        self.x_train = x_train
        self.y_train = y_train
        self.dataset = dataset
        try:
            self.dataset_size = len(self.dataset)
        except:
            self.dataset_size = len(y_train[self.dataset])
        self.batch_size = batch_size
        self.local_round = local_round
        self.dp = dp
        self.DR = DR
        self.DRV2 = DRV2
        self.DRtest = DRtest
        self.Topk = Topk
        self.cpl = cpl
        self.kfilter = kfilter
        self.FLalg = FLalg
        self.rate_dr = rate_dr
        self.grad_norm = grad_norm
        self.grad_perp_norm = grad_perp_norm
        self.lr = lr
        self.momentum = momentum

        self.budget_accountant = budget_accountant
        self.model = None
        self.last_global_model = None
        self.optimizer = None
        self.Vks = None
        self.means = None
        self.is_private = None
        self.global_last_grad = []
        self.device = device
        self.clip_paral = clip_paral
        self.grad_norm_param = self.grad_norm
        self.clipping = ''
        self.k_filter = None

        self.noise = 0
        noise_2 = 0
        if self.dp:
            self.noise = self.budget_accountant.noise_multiplier
            noise_2 = self.budget_accountant.noise_multiplier_2
        if not self.dp and not self.DR and not self.DRV2 and not self.Topk and not self.cpl and not self.kfilter:
            self.grad_norm_param = self.grad_norm
            self.clipping = 'clip_flat'
        # if self.dp or self.Topk or self.DR or self.DRV2 or self.cpl:
        if self.dp and not self.DR and not self.DRV2 and not self.kfilter and not self.cpl:
            self.grad_norm_param = self.grad_norm
            self.clipping = 'flat'
        if not self.dp and self.DR:
            self.grad_norm_param = [self.grad_norm, self.grad_perp_norm, self.rate_dr]
            self.clipping = 'dr_flat'
        if not self.dp and self.DRtest:
            grad_norm = [self.grad_norm, self.grad_perp_norm, noise_2, self.clip_paral]
            if not self.kfilter:
                self.clipping = 'dr_flat_test'
            else:
                self.clipping = 'drkf_flat_test'
        if self.dp and self.DRtest:
            grad_norm = [self.grad_norm, self.grad_perp_norm, noise_2, self.clip_paral]
            if not self.kfilter:
                self.clipping = 'dr_dp_flat_test'
            else:
                self.clipping = 'drkf_dp_flat_test'
        if self.dp and self.DR:
            self.grad_norm_param = [self.grad_norm, self.grad_perp_norm, noise_2]
            self.clipping = 'dr_dp_flat'  
        if self.dp and self.DRV2:
            self.grad_norm_param = [self.grad_norm, self.grad_perp_norm, noise_2]
            self.clipping = 'dr_dp_flat_v2'                
        if self.Topk:
            self.grad_norm_param = [self.grad_norm, self.grad_perp_norm, noise_2, self.rate_dr]
            self.clipping = 'topk_flat'
        if self.dp and self.cpl:
            self.grad_norm_param = [self.grad_norm, self.grad_perp_norm, self.rate_dr]
            self.clipping = 'cpl_dp_flat'    
        if not self.dp and self.cpl:
            self.grad_norm_param = [self.grad_norm, self.grad_perp_norm, self.rate_dr]
            self.clipping = 'cpl_flat'  
        if not self.dp and self.kfilter:
            self.grad_norm_param = [self.grad_norm, self.grad_perp_norm, self.rate_dr]
            self.clipping = 'kfilter_flat' 
            self.k_filter = KalmanFilter([], (self.grad_norm*0.1)**2, (self.grad_perp_norm/self.batch_size)**2)
        if self.dp and self.kfilter:
            self.grad_norm_param = [self.grad_norm, self.grad_perp_norm, self.rate_dr]
            self.clipping = 'kfilter_dp_flat'   
            # self.k_filter = KalmanFilter([], (self.grad_norm*0.1)**2, (0.1*self.grad_norm*self.noise/self.batch_size)**2)
        # test kdp-ref:
        if self.dp:
            self.k_filter = KalmanFilter([], (self.grad_norm*self.noise)**2, (self.grad_norm*self.noise)**2*0.1) 
            
    def download(self, model, global_last_grad):
        self.model = model.to(self.device)
        self.last_global_model = copy.deepcopy(self.model)
        self.global_last_grad = [g.to(self.device) for g in global_last_grad]

    def set_projection(self, Vks=None, means=None, is_private=None):
        self.Vks = Vks
        self.means = means
        self.is_private = is_private

    def precheck(self):
        if not self.budget_accountant:
            return True
        else:
            return self.budget_accountant.precheck(self.dataset_size, self.batch_size, self.local_round)

    def set_optimizer(self):
        pass

    def local_update(self):
        model = self.model.train()
        parameters = model.parameters()

        optimizer = torch.optim.SGD(parameters, lr=self.lr, momentum=self.momentum)
        # optimizer = torch.optim.SGD(parameters, lr=self.lr)
        # if self.DR:
        #     optimizer = torch.optim.SGD(parameters, lr=self.lr, momentum=0.9, weight_decay=0.01)
        criterion = nn.CrossEntropyLoss()

        x_batch = self.x_train[self.dataset]
        y_batch = self.y_train[self.dataset]

        data_batch = TensorDataset(x_batch, y_batch)
        data_loader = DataLoader(data_batch, batch_size=self.batch_size, shuffle=True)

                            
        # print('clipping:', clipping)
        privacy_engine = PrivacyEngine(secure_mode=False)
        model, optimizer, train_loader = privacy_engine.make_private(module=model,
                                                                        optimizer=optimizer,
                                                                        clipping=self.clipping,
                                                                        data_loader=data_loader,
                                                                        noise_multiplier=self.noise,
                                                                        max_grad_norm=self.grad_norm_param) #All of the returned objects act just like their non-private counterparts passed as arguments, but with added DP tasks.


        # global_last_grad
        # if self.DR or self.DRV2 or self.DRtest:
        if self.DR or self.DRV2:
            norm = [p.reshape(-1).norm(2, dim=-1) for p in self.global_last_grad]
            optimizer.last_grad = [p/n for p,n in zip(self.global_last_grad, norm)] 
        if self.Topk:
            norm = [p.reshape(-1).norm(2, dim=-1) for p in self.global_last_grad]
            optimizer.last_grad = [p/n for p,n in zip(self.global_last_grad, norm)] 
            optimizer.last_grad_origin = [p/self.batch_size for p in self.global_last_grad] 
        if self.DRtest:
        # layerwise
            if self.global_last_grad != []:
                last_norm = [p.reshape(-1).norm(2, dim=-1) for p in self.global_last_grad]
                norm = torch.stack(last_norm).norm(2)
                optimizer.norm = norm
                optimizer.last_normratio = [g/norm for g in last_norm]
                # optimizer.last_grad = [p/norm for p in self.global_last_grad] 
                # optimizer.last_normratio = self.global_last_grad
                optimizer.last_grad = [p/n for p,n in zip(self.global_last_grad, last_norm)] 
                optimizer.last_grad_noisy = optimizer.last_grad
            # if self.global_last_grad != []:
                # optimizer.last_grad = self.global_last_grad
                # norm = [p.reshape(-1).norm(2, dim=-1) for p in self.global_last_grad]
                # optimizer.last_grad = [p/n for p,n in zip(self.global_last_grad, norm)] 
            # for KF filter
            self.k_filter.x = optimizer.last_grad
            optimizer.kfilter = self.k_filter
        # if self.kfilter:
            # optimizer.last_grad = [p/self.batch_size for p in self.global_last_grad]
            # self.k_filter.x = optimizer.last_grad
            # optimizer.kfilter = self.k_filter
        if self.dp:
            # test kf-ref
            if self.k_filter.x == []:
                self.k_filter.x = self.global_last_grad
 
        optimizer.global_last_grad = self.global_last_grad # not used temporarily
        logs = []
        # train
        for epoch in range(self.local_round):
            train_acc = 0
            train_loss = 0
            for x_train, y_train in data_loader:
                x_train, y_train = x_train.to(self.device), y_train.to(self.device)

                y_pred = model(x_train)
                loss = criterion(y_pred, y_train)

                _, test_pred = torch.max(y_pred, 1)
                correct = (test_pred == y_train).sum()

                optimizer.zero_grad() # clear grad from last batch
                loss.backward() # back propogation & get gradients
                optimizer.step() # adding noises & update model parameters

                train_acc += correct.item()
                train_loss += loss.item()

                logs.append(copy.deepcopy(optimizer.log))

            # print('Epoch is: %d, Train acc: %.4f, Train loss: %.4f' % ((epoch + 1), train_acc / self.dataset_size, train_loss / self.dataset_size))

        updates = [weight.data for weight in model.state_dict().values()]
        # test kf-ref
        KDP_REF = True
        if KDP_REF == True and self.global_last_grad != []:
            w_delta_noisy = [w - l.data for w,l in zip(updates, self.last_global_model.state_dict().values())]
            self.k_filter.predict()
            w_delta_estimate = self.k_filter.correct(w_delta_noisy)
            updates = [w + l.data for w,l in zip(w_delta_estimate, self.last_global_model.state_dict().values())]
            

        num_parameter1 = 0
        # for u in updates:
        #     num_parameter1 += reduce(mul, u.shape)  # mul对u.shape进行相乘， reduce对这些相乘之后的每个u.shape进行相加

        Bytes1 = num_parameter1 * 4
        # print('num parameters: %d, Bytes: %d, M: %.8f' % (num_parameter1, Bytes1, Bytes1/(1024**2)))

        Bytes2 = logs

        # update the budget accountant
        accum_budget_accountant = self.budget_accountant.update(self.local_round) if self.budget_accountant else None

        return updates, accum_budget_accountant, Bytes1, Bytes2


