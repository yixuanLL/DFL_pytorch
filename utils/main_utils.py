# -*- coding: utf-8 -*-
# @Author : Zhang
# @Email : zl16035056@163.com
# @File : main_utils.py


import os
import csv
import numpy as np
import torch
import random


np.random.seed(10)

def save_progress(args, Accuracy_accountant, Budgets_accountant=None, nbytes1=None, nbytes2=None):
    DATA_MODEL={
        'MNIST': "cnn",
        'CIFAR10': "cnn5",
        'CIFAR100': "resnet",
        'FLamby': "mclr",
        'CAHouse': "mclr"
    }
    eps = 'no-dp'
    if args.dp:
        if args.DR or args.DRV2 or args.DRtest:
            eps = args.eps+args.eps_2
        else:
            eps = args.eps

    save_dir = os.path.join(os.getcwd(), args.save_dir, 'result', args.dataset,
                            # ('noniid' if args.noniid else 'iid'),
                            # DATA_MODEL[args.dataset],
                            str(args.num_clients),
                            (str(eps) if args.dp else 'no-dp'))

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    file_name = '{}{}{}{}{}{}{}{}{}{}{}{}{}'.format(args.lr,
                              ('-'+str(args.opt)),
                              ('-'+str(args.momentum)),
                              ('-DR' if args.DR else ''),
                              ('-DRV2' if args.DRV2 else ''),
                              ('-DRtest' if args.DRtest else ''),
                              ('-cpl' if args.cpl else ''),
                              ('-SGD' if not args.DR and not args.DRV2 and not args.DRtest and not args.cpl else ''),
                              ('-'+str(args.eps) if args.dp else '-0'),
                              ('-'+str(args.eps_2) if args.DR or args.DRV2 or args.DRtest else '-0'),
                              ('-'+str(args.grad_norm)),
                              ('-'+str(args.grad_perp_norm)),
                              ('-'+str(args.clip_paral)))
                            #   ('-'+str(args.global_round)))

    with open(os.path.join(save_dir, file_name + '.csv'), 'w') as file:
        writer = csv.writer(file, delimiter=',')
        if args.dp:
            writer.writerow(Budgets_accountant)
        if args.DR and args.dp:
            writer.writerow(nbytes1)
            writer.writerow(nbytes2)

        writer.writerow(Accuracy_accountant)
    path = os.path.join(save_dir, file_name + '.csv')
    # print(path)
    return path


def print_accuracy_and_loss(r, test_accuracy, test_loss):
    # print('-------------------------------------------------------------------------------------')
    print('round %d global model has test acc: %.4f  test loss: %.4f' % (r, test_accuracy, test_loss))
    # print('-------------------------------------------------------------------------------------')


def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True