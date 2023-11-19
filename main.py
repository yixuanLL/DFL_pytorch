# -*- coding: utf-8 -*-
# @Author : yixuan
# @File : main.py


import copy
import torch
import argparse
import importlib
from utils.create_dataset import prepare_local_dataset
from utils.dataloader import loader
# from models.client_kdp import Client
# print('__client_kdp__')
# from models.client import Client
# print('__client__')
# from models.server import Server
# from utils.dpsgd_utils import compute_noise_multiplier
# from utils.budgets_accountant import BudgetsAccountant
# from utils.main_utils import save_progress, print_accuracy_and_loss, setup_seed
# import os
# from utils.grad_plot import grad_plot, grad_var, grad_var_t, loss_plot, grad_plot_t


from torchvision import datasets, transforms
# os.environ['CUDA_VISIBLE_DEVICES'] ='0'
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PARAMS={
    'MNIST': (784,10),
    'CIFAR10': (3*32*32,10),
    'FLamby': (13,2)
}

def main():
    accuracy_accountant = []
    # privacy_accountant = []
    # accum_nbytes_list1 = []
    # accum_nbytes_list2 = []
    # accum_nbytes1 = 0
    # accum_nbytes2 = 0
    # max_accum_budget_accountant = 0
    # save_address = ''
    # set seed
    # setup_seed(args.seed)
    # prepare local dataset
    # train_dataloader = datasets.MNIST(root='~/data', train=True, download=False, transform=transforms.ToTensor())
    # x_train = train_dataloader.data.float().unsqueeze(1)#.to(device)
    # x_train, y_train, x_test, y_test = loader(args.dataset, args.noniid)
 
if __name__ == '__main__':
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--save_dir', type=str, default='result')
    # parser.add_argument('--dataset', type=str, default='MNIST')
    # parser.add_argument('--FLalg', type=str, default='FedAvg', help='Algorithm of FL')
    # parser.add_argument('--DR', type=bool, default=False)
    # parser.add_argument('--DRV2', type=bool, default=False)
    # parser.add_argument('--DRtest', type=bool, default=False)
    # parser.add_argument('--global_round', type=int, default=100)
    # parser.add_argument('--local_round', type=int, default=2)
    # parser.add_argument('--noniid', type=bool, default=False, help='if True, use noniid data')
    # parser.add_argument('--num_clients', type=int, default=2) 
    # parser.add_argument('--batch_size', type=int, default=128)
    # parser.add_argument('--dp', type=bool, default=False, help='if True, use differential privacy')
    # parser.add_argument('--eps', type=float, default=1)
    # parser.add_argument('--eps_2', type=float, default=0.02)
    # parser.add_argument('--delta', type=float, default=1e-5, help='differential privacy parameter')
    # parser.add_argument('--grad_norm', type=float, default=10)
    # parser.add_argument('--grad_perp_norm', type=float, default=0.2)
    # parser.add_argument('--sample_ratio', type=float, default=1)
    # parser.add_argument('--seed', type=int, default=0)
    # parser.add_argument('--model', type=str, default='cnn')
    # parser.add_argument('--lr', type=float, default=0.2)
    # parser.add_argument('--momentum', type=float, default=0.)
    # parser.add_argument('--Topk', type=bool, default=False)
    # parser.add_argument('--cpl', type=bool, default=False)
    # parser.add_argument('--kf', type=bool, default=False)
    # parser.add_argument('--rate_dr', type=float, default=1, help='sparse rate in directional reduction')
    # parser.add_argument('--clip_paral', type=float, default=0.01, help='parallel alpha bound')
    # args = parser.parse_args() 

    # # print arguments
    # try: parsed = vars(parser.parse_args())
    # except IOError as msg: parser.error(str(msg))
    # maxLen = max([len(ii) for ii in parsed.keys()]);
    # fmtString = '\t%' + str(maxLen) + 's : %s';
    # print('Arguments:')
    # for keyPair in parsed.items(): print(fmtString % keyPair)
    

    train_dataloader = datasets.MNIST(root='~/data', train=True, download=False, transform=transforms.ToTensor())
    x_train = train_dataloader.data.float().unsqueeze(1)#.to(device)
    # main(args)
    # main()
    print('[FINISHED]')