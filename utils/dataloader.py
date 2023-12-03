# -*- coding: utf-8 -*-
# @Author : yixuan
# @Email : zl16035056@163.com
# @File : dataloader.py


import torch
from torchvision import datasets, transforms
import numpy as np
from flamby.datasets.fed_heart_disease import HeartDiseaseRaw, FedHeartDisease

def loader(name, noniid):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = 'cuda'
    if name == 'MNIST':
        train_dataloader = datasets.MNIST(root='~/data', train=True, download=False, transform=transforms.ToTensor())
        x_train = train_dataloader.data.float().unsqueeze(1)
        y_train = train_dataloader.targets
        # normalize manually, as "transform.totensor" does not work well
        me = 0.1307
        std = 0.3081
        indices_train = torch.argsort(y_train)
        sorted_x_train = x_train[indices_train]
        # sorted_x_train = (x_train[indices_train] / 255. - me) / std

        sorted_y_train = y_train[indices_train]
        test_dataloader = datasets.MNIST(root='~/data', train=False, download=False, transform=transforms.ToTensor())
        # x_test = (test_dataloader.data.float().unsqueeze(1) / 255. - me) / std
        x_test = test_dataloader.data.float().unsqueeze(1)
        y_test = test_dataloader.targets
        

    if name == 'CIFAR10':
        transform = transforms.Compose(
            [transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        train_dataloader = datasets.CIFAR10(root='~/data', train=True, download=False, transform=transform)
        train_data = torch.utils.data.DataLoader(train_dataloader, batch_size=50000, shuffle=False, num_workers=0)
        x_train, y_train  = next(iter(train_data))
        
        indices_train = torch.argsort(y_train)
        sorted_x_train = x_train[indices_train]
        sorted_y_train = y_train[indices_train]

        test_dataloader = datasets.CIFAR10(root='~/data', train=False, download=False, transform=transform)  
        train_data = torch.utils.data.DataLoader(test_dataloader, batch_size=10000, shuffle=False, num_workers=0)
        x_test, y_test  = next(iter(train_data))      

    if name == 'FLamby':
        if not noniid:
            traindata_pooled = FedHeartDisease(train=True, pooled=True)
            train_data = torch.utils.data.DataLoader(traindata_pooled, batch_size=486, shuffle=False, num_workers=0)
            x_train, y_train  = next(iter(train_data))
            # normalize & standarlize
            # mi = torch.min(x_train)
            # ma = torch.max(x_train)
            # x_train = (x_train - mi) / (ma-mi)
            # mu = torch.mean(x_train)
            # std = torch.std(x_train)
            # x_train = (x_train - mu) / std

            y_train = y_train.reshape(486,).to(torch.int64)
            indices_train = torch.argsort(y_train)
            sorted_x_train = x_train[indices_train]
            sorted_y_train = y_train[indices_train]

            testdata_pooled = FedHeartDisease(train=False, pooled=True)
            test_data = torch.utils.data.DataLoader(testdata_pooled, batch_size=254, shuffle=False, num_workers=0)
            x_test, y_test  = next(iter(test_data)) 
            y_test = y_test.reshape(254,).to(torch.int64)
    
        else:
            sorted_x_train = []
            sorted_y_train = []
            x_list = []
            y_list = []
            client_set = [0, 1, 2, 3]
            for center in client_set:
                traindata = FedHeartDisease(train=True, pooled=False, center=center)
                traindata_size = len(traindata)
                train_data = torch.utils.data.DataLoader(traindata, batch_size=traindata_size, shuffle=False, num_workers=0)
                x_train, y_train  = next(iter(train_data))
                y_train = y_train.reshape(traindata_size,).to(torch.int64)
                sorted_x_train.append(x_train)
                sorted_y_train.append(y_train)

                testdata = FedHeartDisease(train=False, pooled=False, center=center)
                testdata_size = len(testdata)
                test_data = torch.utils.data.DataLoader(testdata, batch_size=testdata_size, shuffle=False, num_workers=0)
                x, y  = next(iter(test_data))
                x_list.append(x)
                y_list.append(y.reshape(testdata_size,).to(torch.int64))
            x_test = torch.cat(x_list, dim=0)
            y_test = torch.cat(y_list)

            labels_set = []
            # for cid in range(N):
            for cid in range(len(client_set)):
                # idx = [int(val) for val in client_set[cid]]
                labels_set.append(set(np.array(sorted_y_train[cid])))

                labels_count = [0]*2
                for label in np.array(sorted_y_train[cid]):
                    labels_count[int(label)] += 1
                print('cid: {}, number of labels: {}/2.'.format(cid, len(labels_set[cid])))
                print(labels_count)
            print()
            
            

    print('Using {} dataset!\n'.format(name))


    return sorted_x_train, sorted_y_train, x_test, y_test