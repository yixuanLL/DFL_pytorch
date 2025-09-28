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
from opacus.utils.batch_memory_manager import BatchMemoryManager

class Client(nn.Module):
    def __init__(self, x_train, y_train, x_test, y_test, dataset, dataname, batch_size, FLalg, dp, DR, DRV2, DRtest,Topk, cpl, kfilter, rate_dr, local_round, grad_norm, grad_perp_norm, lr, momentum, budget_accountant, device, opt, alg, num_clients, clip_paral, steps_interval, steps_dr, weight_alpha):
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
        self.alg = alg
        self.steps_interval = steps_interval
        self.steps_dr = steps_dr
        self.weight_alpha = weight_alpha

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
        noise_g = 0
        noise_p = 0
        noise_a = 0
        if self.dp:
            noise_g = self.budget_accountant.noise_multiplier_g
            noise_p = self.budget_accountant.noise_multiplier_p
            noise_a = self.budget_accountant.noise_multiplier_a
            noise = noise_g
        if self.alg == 'autoclip':
            grad_norm = self.grad_norm
            clipping = 'autoclip_flat'
        if self.dp and not self.DR and not self.DRV2 and not self.DRtest and not self.cpl:
            grad_norm = self.grad_norm
            clipping = 'flat'
        if not self.dp and self.DR:
            grad_norm = [self.grad_norm, self.grad_perp_norm, 0.0, self.clip_paral]
            clipping = 'dr_flat'
            noise = 0.0
        if not self.dp and self.DRtest:
            grad_norm = [self.grad_norm, self.grad_perp_norm, 0.0, self.clip_paral]
            clipping = 'dr_flat_test'
            noise = 0.0
        if self.dp and self.DRtest:
            grad_norm = [self.grad_norm, self.grad_perp_norm, noise_a, self.clip_paral]
            clipping = 'dr_dp_flat_test'
            noise = noise_p
        if self.dp and self.DR: # for DRV7
            grad_norm = [self.grad_norm, self.grad_perp_norm, noise_a, self.clip_paral, noise_g, self.steps_interval, self.steps_dr, self.weight_alpha]
            clipping = 'dr_dp_flat' 
            noise = noise_p 
        if self.dp and self.DRV2:
            grad_norm = [self.grad_norm, self.grad_perp_norm, noise_a]
            clipping = 'dr_dp_flat_v2'  
            noise = noise_g              
        if self.Topk:
            grad_norm = [self.grad_norm, self.grad_perp_norm, noise_a, self.rate_dr]
            clipping = 'topk_flat'
        if self.dp and self.cpl:
            grad_norm = [self.grad_norm, self.grad_perp_norm, self.rate_dr]
            clipping = 'cpl_dp_flat'   
            noise = noise_g 
        if not self.dp and self.cpl:
            grad_norm = [self.grad_norm, self.grad_perp_norm, self.rate_dr]
            clipping = 'cpl_flat'  
            noise = 0.0
                              
        # print('clipping:', clipping)
        if self.dp or self.DR or self.DRtest or self.DRV2:
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
                optimizer.last_grad = [p/(n+1e-8) for p,n in zip(self.global_last_grad, last_norm)] # for 2024/01 result
                # optimizer.last_grad = [p/norm for p in self.global_last_grad] # opt 1
                # optimizer.last_grad = self.global_last_grad # opt 2
                optimizer.last_grad_noisy = optimizer.last_grad

 
        optimizer.global_last_grad = self.global_last_grad # not used temporarily
        logs = []
        losses = []
        accs =[]
        gl = None
        # train
        for epoch in range(self.local_round):
            train_acc = 0
            train_loss = 0
            with BatchMemoryManager(
                    data_loader=train_loader, 
                    max_physical_batch_size=32, 
                    optimizer=optimizer
                ) as memory_safe_data_loader:
                for i, (x_train, y_train) in enumerate(memory_safe_data_loader):
            # if True: #original batch size, we use it for observing 
            #     for i, (x_train, y_train) in enumerate(data_loader):
                    x_train, y_train = x_train.to(self.device), y_train.to(self.device)

                    y_pred = model(x_train)
                    loss = criterion(y_pred, y_train)

                    _, test_pred = torch.max(y_pred, 1)
                    correct = (test_pred == y_train).sum()

                    optimizer.zero_grad() # clear grad from last batch
                    loss.backward() # back propogation & get gradients

                    optimizer.step() # adding noises & update model parameters
                    
                    # else:
                    #     optimizer.virtual_step()                    
                    optimizer.steps += 1
                    train_acc += correct.item()
                    train_loss += loss.item()
                    
                    # if optimizer.steps % 50 == 1: # log save per_sample
                    if True: # log save mean
                        logs.append(copy.deepcopy(optimizer.log))
                    # for SVHN
                    # if len(optimizer.log) > 1:

                    #     if i==0 and epoch==0:
                    #         gdn = 0
                    #     else:
                    #         gd = [gi-gli for gi, gli in zip(optimizer.log[0], gl)]
                    #         gdn = torch.norm(grad_flat(gd))
                    #     gg = grad_flat(optimizer.log[0])
                    #     gn = torch.norm(gg)
                        
                    #     gl = copy.deepcopy(optimizer.log[0])
                    #     logs.append([gn, gdn, 0])



                # losses.append(copy.deepcopy(train_loss)/self.dataset_size)
                if self.num_clients == 1:
                    test_acc, test_loss = self.test(copy.deepcopy(model))
                    print('Epoch is: %d, Train acc: %.4f, Train loss: %.4f, Test acc: %.4f, Test loss: %.4f' % ((epoch + 1), train_acc / self.dataset_size, train_loss / self.dataset_size, test_acc, test_loss))
                    accs.append(test_acc)
                # print('Epoch is: %d, Train acc: %.4f, Train loss: %.4f' % ((epoch + 1), train_acc / self.dataset_size, train_loss / self.dataset_size))

        updates = [weight.data for weight in model.state_dict().values()]
        num_parameter1 = 0
        # for u in updates:
        #     num_parameter1 += reduce(mul, u.shape)  # mul对u.shape进行相乘， reduce对这些相乘之后的每个u.shape进行相加

        Bytes1 = num_parameter1 * 4
        # print('num parameters: %d, Bytes: %d, M: %.8f' % (num_parameter1, Bytes1, Bytes1/(1024**2)))

        Bytes2 = (logs, losses, accs)

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