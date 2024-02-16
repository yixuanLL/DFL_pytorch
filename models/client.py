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
# from torchmetrics.functional.regression import mean_squared_error

class Client(nn.Module):
    def __init__(self, x_train, y_train, x_test, y_test, dataset, dataname, batch_size, FLalg, dp, DR, DRV2, DRtest,Topk, cpl, kfilter, rate_dr, local_round, grad_norm, grad_perp_norm, lr, momentum, budget_accountant, device, opt, num_clients, clip_paral):
        super(Client, self).__init__()
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.dataset = dataset
        self.dataname = dataname

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
        self.opt = opt
        self.num_clients = num_clients

    def download(self, model, global_last_grad):
        # self.model = model
        # self.global_last_grad = global_last_grad
        self.model = model.to(self.device)
        self.global_last_grad = [g.to(self.device) for g in global_last_grad] #OOM

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
        # parameters = model.parameters()
        # default optimizer: SGD
        optimizer = torch.optim.SGD(model.nn_layer.parameters(), lr=self.lr, momentum=self.momentum)
        if self.opt == 'rmsprop':
            optimizer = torch.optim.RMSprop(model.nn_layer.parameters(), lr=self.lr)
        if self.opt == 'adam':
            optimizer = torch.optim.Adam(model.nn_layer.parameters(), lr=self.lr)

        if self.dataname == 'CAHouse':
            criterion = nn.MSELoss(reduction='sum')
        else:
            criterion = nn.CrossEntropyLoss()

        x_batch = self.x_train[self.dataset]
        y_batch = self.y_train[self.dataset]

        data_batch = TensorDataset(x_batch, y_batch)
        data_loader = DataLoader(data_batch, batch_size=self.batch_size, shuffle=True)
        seed = 0
        # torch.manual_seed(seed)
        noise = 0
        noise_2 = 0
        if self.dp:
            noise = self.budget_accountant.noise_multiplier
            noise_2 = self.budget_accountant.noise_multiplier_2
        if not self.dp and not self.DR and not self.DRV2 and not self.Topk and not self.cpl and not self.kfilter:
            grad_norm = self.grad_norm
            # clipping = 'clip_flat' #不需要单独验证clip的效果了
            clipping = 'flat'
        # if self.dp or self.Topk or self.DR or self.DRV2 or self.cpl:
        if self.dp and not self.DR and not self.DRV2 and not self.kfilter and not self.cpl:
            grad_norm = self.grad_norm
            clipping = 'flat'
        if not self.dp and self.DR:
            grad_norm = [self.grad_norm, self.grad_perp_norm, noise_2, self.clip_paral]
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
        if self.dp and self.DR: # for DRV5
            noise_3 = self.budget_accountant.noise_multiplier_3
            grad_norm = [self.grad_norm, self.grad_perp_norm, noise_2, self.clip_paral, noise_3]
            clipping = 'dr_dp_flat'  
        if self.dp and self.DRV2:
            grad_norm = [self.grad_norm, self.grad_perp_norm, noise_2]
            clipping = 'dr_dp_flat_v2'                
        if self.Topk:
            grad_norm = [self.grad_norm, self.grad_perp_norm, noise_2, self.rate_dr]
            clipping = 'topk_flat'
        if self.dp and self.cpl:
            grad_norm = [self.grad_norm, self.grad_perp_norm, self.rate_dr]
            if not self.kfilter:
                clipping = 'cpl_dp_flat'    
            else:
                clipping = 'clip_kf_dp_flat'
        if not self.dp and self.cpl:
            grad_norm = [self.grad_norm, self.grad_perp_norm, self.rate_dr]
            clipping = 'cpl_flat'  
        if not self.dp and self.kfilter and not self.DRtest:
            grad_norm = [self.grad_norm, self.grad_perp_norm, self.rate_dr]
            clipping = 'kfilter_flat' 
        if self.dp and self.kfilter and not self.DRtest:
            grad_norm = [self.grad_norm, self.grad_perp_norm, self.rate_dr]
            clipping = 'kfilter_dp_flat'                               
        # print('clipping:', clipping)
        if self.dp:
            privacy_engine = PrivacyEngine(secure_mode=False)
            model, optimizer, train_loader = privacy_engine.make_private(module=model,
                                                                        optimizer=optimizer,
                                                                        clipping=clipping,
                                                                        data_loader=data_loader,
                                                                        noise_multiplier=noise,
                                                                        max_grad_norm=grad_norm) #All of the returned objects act just like their non-private counterparts passed as arguments, but with added DP tasks.


        # global_last_grad
        # if self.DR or self.DRV2 or self.DRtest:
        # if self.DR or self.DRV2:
        if False:
            norm = [p.reshape(-1).norm(2, dim=-1) for p in self.global_last_grad]
            optimizer.last_grad = [p/n for p,n in zip(self.global_last_grad, norm)] 
        if self.Topk:
            norm = [p.reshape(-1).norm(2, dim=-1) for p in self.global_last_grad]
            optimizer.last_grad = [p/n for p,n in zip(self.global_last_grad, norm)] 
            optimizer.last_grad_origin = [p/self.batch_size for p in self.global_last_grad] 
        if self.DRtest or self.DR or self.cpl:
            if self.global_last_grad != []:
                last_norm = [p.reshape(-1).norm(2, dim=-1) for p in self.global_last_grad]
                norm = torch.stack(last_norm).norm(2)
                optimizer.norm = norm
                optimizer.last_normratio = [g/norm for g in last_norm]
                optimizer.last_grad = [p/n for p,n in zip(self.global_last_grad, last_norm)] 
                # optimizer.last_grad = [p/norm for p in self.global_last_grad] # opt 1
                # optimizer.last_grad = self.global_last_grad # opt 2
                optimizer.last_grad_noisy = optimizer.last_grad
            # for KF filter
            if self.kfilter:
                if self.dp:
                    optimizer.kfilter = KalmanFilter(optimizer.last_grad, (self.grad_norm*0.1)**2, (0.1*self.grad_perp_norm*noise/self.batch_size)**2)
                else:
                    optimizer.kfilter = KalmanFilter(optimizer.last_grad, (self.grad_norm*0.1)**2, (self.grad_perp_norm*0.1)**2)
        if self.kfilter and not self.DRtest:
            optimizer.last_grad = [p/self.batch_size for p in self.global_last_grad]
            # entire gradient filter
            optimizer.kfilter = KalmanFilter(optimizer.last_grad, (self.grad_norm*0.1)**2, (0.1*self.grad_norm*noise/self.batch_size)**2)

 
        optimizer.global_last_grad = self.global_last_grad # not used temporarily
        logs = []
        losses = []
        
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
                
                optimizer.steps += 1
                train_acc += correct.item()
                train_loss += loss.item()
                
                # logs.append(copy.deepcopy(optimizer.log))
                # self.longlogs.append(copy.deepcopy(optimizer.log))
            # losses.append(copy.deepcopy(train_loss)/self.dataset_size)
            if self.num_clients == 1:
                test_acc, test_loss = self.test(copy.deepcopy(model))
                print('Epoch is: %d, Train acc: %.4f, Train loss: %.4f, Test acc: %.4f, Test loss: %.4f' % ((epoch + 1), train_acc / self.dataset_size, train_loss / self.dataset_size, test_acc, test_loss))
            
            # print('Epoch is: %d, Train acc: %.4f, Train loss: %.4f' % ((epoch + 1), train_acc / self.dataset_size, train_loss / self.dataset_size))

        updates = [weight.data for weight in model.state_dict().values()]
        if self.FLalg == 'FedDrAvg_upload': # upload gi_perp costheta
            updates = [optimizer.g_perp_sum, optimizer.cos_sum]
        num_parameter1 = 0
        # for u in updates:
        #     num_parameter1 += reduce(mul, u.shape)  # mul对u.shape进行相乘， reduce对这些相乘之后的每个u.shape进行相加

        Bytes1 = num_parameter1 * 4
        # print('num parameters: %d, Bytes: %d, M: %.8f' % (num_parameter1, Bytes1, Bytes1/(1024**2)))

        Bytes2 = (logs, losses)

        # update the budget accountant
        accum_budget_accountant = self.budget_accountant.update(self.local_round) if self.budget_accountant else None

        return updates, accum_budget_accountant, Bytes1, Bytes2

    def test(self, model):
        model.eval() #.to(self.device)
        data_loader = TensorDataset(self.x_test.to(self.device), self.y_test.to(self.device))
        data_loader = DataLoader(data_loader, batch_size=128, shuffle=True)
        if self.dataname == 'CAHouse':
            criterion = nn.MSELoss()
        else:
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