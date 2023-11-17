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
from utils.kalman_filter import KalmanFilter, KalmanFilterLayer
import copy
from utils.grad_plot import grad_flat

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
        self.optimizer = None
        self.Vks = None
        self.means = None
        self.is_private = None
        self.global_last_grad = []
        self.device = device
        self.clip_paral = clip_paral
        self.noisy_layervar = [0.3, 0.3]
        self.longlogs = []
        self.ratio = 1
    def download(self, model, global_last_grad):
        self.model = model.to(self.device)
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

        noise = 0
        noise_2 = 0
        if self.dp:
            noise = self.budget_accountant.noise_multiplier
            noise_2 = self.budget_accountant.noise_multiplier_2
        if not self.dp and not self.DR and not self.DRV2 and not self.Topk and not self.cpl and not self.kfilter:
            grad_norm = self.grad_norm
            clipping = 'clip_flat'
        # if self.dp or self.Topk or self.DR or self.DRV2 or self.cpl:
        if self.dp and not self.DR and not self.DRV2 and not self.kfilter and not self.cpl:
            grad_norm = self.grad_norm
            clipping = 'flat'
        if not self.dp and self.DR:
            grad_norm = [self.grad_norm, self.grad_perp_norm, self.rate_dr]
            clipping = 'dr_flat'
        if not self.dp and self.DRtest:
            grad_norm = [self.grad_norm, self.grad_perp_norm, noise_2, self.clip_paral]
            if not self.kfilter:
                clipping = 'dr_flat_test'
            else:
                clipping = 'drkf_flat_test'
        if self.dp and self.DRtest:
            grad_norm = [self.grad_norm, self.grad_perp_norm, noise_2, self.clip_paral]
            if not self.kfilter:
                clipping = 'dr_dp_flat_test'
            else:
                clipping = 'drkf_dp_flat_test'
        if self.dp and self.DR:
            grad_norm = [self.grad_norm, self.grad_perp_norm, noise_2]
            clipping = 'dr_dp_flat'  
        if self.dp and self.DRV2:
            grad_norm = [self.grad_norm, self.grad_perp_norm, noise_2]
            clipping = 'dr_dp_flat_v2'                
        if self.Topk:
            grad_norm = [self.grad_norm, self.grad_perp_norm, noise_2, self.rate_dr]
            clipping = 'topk_flat'
        if self.dp and self.cpl:
            grad_norm = [self.grad_norm, self.grad_perp_norm, self.rate_dr]
            clipping = 'cpl_dp_flat'    
        if not self.dp and self.cpl:
            grad_norm = [self.grad_norm, self.grad_perp_norm, self.rate_dr]
            clipping = 'cpl_flat'  
        if not self.dp and self.kfilter:
            grad_norm = [self.grad_norm, self.grad_perp_norm, self.rate_dr]
            clipping = 'kfilter_flat' 
        if self.dp and self.kfilter:
            grad_norm = [self.grad_norm, self.grad_perp_norm, self.rate_dr]
            clipping = 'kfilter_dp_flat'                               
        # print('clipping:', clipping)
        privacy_engine = PrivacyEngine(secure_mode=False)
        model, optimizer, train_loader = privacy_engine.make_private(module=model,
                                                                        optimizer=optimizer,
                                                                        clipping=clipping,
                                                                        data_loader=data_loader,
                                                                        noise_multiplier=noise,
                                                                        max_grad_norm=grad_norm) #All of the returned objects act just like their non-private counterparts passed as arguments, but with added DP tasks.


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
            if self.global_last_grad != []:
                last_norm = [p.reshape(-1).norm(2, dim=-1) for p in self.global_last_grad]
                norm = torch.stack(last_norm).norm(2)
                optimizer.norm = norm
                optimizer.last_normratio = [g/norm for g in last_norm]
                optimizer.last_grad = [p/n for p,n in zip(self.global_last_grad, last_norm)] 
                optimizer.last_grad_noisy = optimizer.last_grad
            # for KF filter
            if self.kfilter:
                if self.dp:
                    optimizer.kfilter = KalmanFilter(optimizer.last_grad, (self.grad_norm*0.1)**2, (self.grad_perp_norm*noise/self.batch_size)**2)
                else:
                    optimizer.kfilter = KalmanFilter(optimizer.last_grad, (self.grad_norm*0.1)**2, (self.grad_perp_norm*0.1)**2)
        if self.kfilter:
            optimizer.last_grad = [p/self.batch_size for p in self.global_last_grad]
            # entire gradient filter
            optimizer.kfilter = KalmanFilter(optimizer.last_grad, (self.grad_norm*0.1)**2, (0.1*self.grad_norm*noise/self.batch_size)**2)
            # layer-wise gradient filter
            # grad_noise_var = [(self.grad_norm*0.01)**2, (self.grad_norm*0.1)**2]
            # grad_noise_var = [(self.grad_norm*noise/self.batch_size)**2 * 0.001, (self.grad_norm*noise/self.batch_size)**2 * 0.01]
            # grad_noise_var = [0.00005, 0.0006]
            # dp_noise_var = (self.grad_norm*noise/self.batch_size)**2 * 0.001

            # grad_noise_var = [0.00005+0.5, 0.0006+0.5]
            # grad_noise_var = [0.5,0.5]
            # obsv_noise_var = [0.05, 0.1]

            # grad_noise_var = self.noisy_layervar
            # true_g = [0.00005, 0.006]
            # obsv_noise_var = [v-(self.grad_norm*noise/self.batch_size)**2-e for v,e in zip(self.noisy_layervar, true_g)]
            # obsv_noise_var = [0.05, 0.05]

            # grad_noise_var = self.noisy_layervar
            # n_var= (self.grad_norm*noise/self.batch_size)**2
            # grad_noise_var = [n_var+0.005, n_var+0.06]
            # obsv_noise_var = []
            # if self.noisy_layervar != []:
            #     obsv_noise_var = [abs(self.noisy_layervar[0]-n_var), abs(self.noisy_layervar[1]-n_var)]
            # # obsv_noise_var = [abs(g-n_var)*self.ratio for g in self.noisy_layervar]
            # optimizer.kfilter = KalmanFilterLayer(optimizer.last_grad, grad_noise_var, obsv_noise_var)

            # grad_noise_var = self.noisy_layervar
            # n_var= (self.grad_norm*noise/self.batch_size)**2
            # grad_noise_var = [0.005, 0.06]
            # # grad_noise_var = [n_var, n_var]
            # obsv_noise_var = []
            # if self.noisy_layervar != []:
            #     obsv_noise_var = [abs(self.noisy_layervar[0]-0.005), abs(self.noisy_layervar[1]-0.06)]
            # # obsv_noise_var = [abs(g-n_var)*self.ratio for g in self.noisy_layervar]
            # optimizer.kfilter = KalmanFilterLayer(optimizer.last_grad, grad_noise_var, obsv_noise_var)
 
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
                self.longlogs.append(copy.deepcopy(optimizer.log))

            # print('Epoch is: %d, Train acc: %.4f, Train loss: %.4f' % ((epoch + 1), train_acc / self.dataset_size, train_loss / self.dataset_size))
        if self.kfilter:
            # var of noisy gradients
            layer_num = [len(g.reshape(-1)) for g in optimizer.last_grad]
            noisy = []
            self.noisy_layervar = []
            for i in range(len(self.longlogs)): #round
                _, n, _ = self.longlogs[i]
                noisy.append(grad_flat(n))
            noisy =  torch.var(torch.stack(noisy, dim=0), dim=0)
            start = 0
            for num in layer_num:
                self.noisy_layervar.append( torch.mean(noisy[start:start+num]))
                start += num

        updates = [weight.data for weight in model.state_dict().values()]
        if self.FLalg == 'FedDrAvg_upload': # upload gi_perp costheta
            updates = [optimizer.g_perp_sum, optimizer.cos_sum]
        num_parameter1 = 0
        # for u in updates:
        #     num_parameter1 += reduce(mul, u.shape)  # mul对u.shape进行相乘， reduce对这些相乘之后的每个u.shape进行相加

        Bytes1 = num_parameter1 * 4
        # print('num parameters: %d, Bytes: %d, M: %.8f' % (num_parameter1, Bytes1, Bytes1/(1024**2)))

        Bytes2 = logs

        # update the budget accountant
        accum_budget_accountant = self.budget_accountant.update(self.local_round) if self.budget_accountant else None

        return updates, accum_budget_accountant, Bytes1, Bytes2

