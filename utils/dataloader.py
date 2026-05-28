# -*- coding: utf-8 -*-
# @Author : yixuan
# @Email : zl16035056@163.com
# @File : dataloader.py


import torch
from torchvision import datasets, transforms
import numpy as np
# from sklearn.datasets import fetch_california_housing
# from sklearn.model_selection import train_test_split
# from torchtext import datatxt
# from torchtext import datasetstxt
# from torchtext.vocab import GloVe
import string

# def tokenize(input):
#     """
#         Naive tokenizer, that lower-cases the input
#         and splits on punctuation and whitespace
#     """
#     input = input.lower()
#     for p in string.punctuation:
#         input = input.replace(p," ")
#     return input.strip().split()


# def num2words(vocab,vec):
#     """
#         Converts a vector of word indicies
#         to a list of strings
#     """
#     return [vocab.itos[i] for i in vec]

def loader(name, noniid):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # if name == 'IMDB':
    #     TEXT = datatxt.Field(lower=True, include_lengths=True, batch_first=True,tokenize=tokenize,fix_length=max_length)
    #     LABEL = datatxt.Field(sequential=False,unk_token=None,pad_token=None)
    #     train, test = datasetstxt.IMDB.splits(TEXT, LABEL)
    #     print(train.shape, test.shape)

    # if name == 'CAHouse':
    #     CA = fetch_california_housing()
    #     # feature_names = CA.feature_names
    #     #take first n examples for speed up
    #     # n = 600
    #     X = CA.data #[:n]
    #     y = CA.target #[:n]
    #     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
    #     x_train = torch.tensor(X_train).float()
    #     y_train = torch.tensor(y_train).view(-1, 1).float()

    #     x_test = torch.tensor(X_test).float()
    #     y_test = torch.tensor(y_test).view(-1, 1).float()

    #     # data = torch.utils.data.TensorDataset(X_train, y_train)
    #     # train_iter = torch.utils.data.DataLoader(data, batch_size=10, shuffle=True)
    #     # pass
    #     # indices_train = torch.argsort(y_train)
    #     sorted_x_train = x_train #[indices_train]
    #     sorted_y_train = y_train #[indices_train]
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
        test_data = torch.utils.data.DataLoader(test_dataloader, batch_size=10000, shuffle=False, num_workers=0)
        x_test, y_test  = next(iter(test_data))  
    
    if name == 'SVHN':
        T_normalize = transforms.Normalize(mean = [0.485, 0.456, 0.406],std = [0.229, 0.224, 0.225])
        transformation = transforms.Compose([transforms.RandomHorizontalFlip(),  transforms.ToTensor(), T_normalize])  
        train_dataloader = datasets.SVHN(root='~/data', split='train', download=False, transform=transformation)  
        train_data = torch.utils.data.DataLoader(train_dataloader, batch_size=73257, shuffle=False, num_workers=0)
        x_train, y_train  = next(iter(train_data))

        indices_train = torch.argsort(y_train)
        sorted_x_train = x_train[indices_train] #[:256]
        sorted_y_train = y_train[indices_train] #[:256]

        test_dataloader = datasets.SVHN(root='~/data', split='test', download=False, transform=transformation)  
        test_data = torch.utils.data.DataLoader(test_dataloader, batch_size=27000, shuffle=False, num_workers=0)
        x_test, y_test  = next(iter(test_data))  

    if name == 'CIFAR100':
        transform = transforms.Compose(
            [transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        train_dataloader = datasets.CIFAR100(root='~/data', train=True, download=False, transform=transform)
        train_data = torch.utils.data.DataLoader(train_dataloader, batch_size=50000, shuffle=False, num_workers=0)
        x_train, y_train  = next(iter(train_data))
        
        indices_train = torch.argsort(y_train)
        sorted_x_train = x_train[indices_train]
        sorted_y_train = y_train[indices_train]
        test_dataloader = datasets.CIFAR100(root='~/data', train=False, download=False, transform=transform)  
        train_data = torch.utils.data.DataLoader(test_dataloader, batch_size=10000, shuffle=False, num_workers=0)
        x_test, y_test  = next(iter(train_data))      

    # if name == 'FLamby':
    #     if not noniid:
    #         traindata_pooled = FedHeartDisease(train=True, pooled=True)
    #         train_data = torch.utils.data.DataLoader(traindata_pooled, batch_size=486, shuffle=False, num_workers=0)
    #         x_train, y_train  = next(iter(train_data))
    #         # normalize & standarlize
    #         # mi = torch.min(x_train)
    #         # ma = torch.max(x_train)
    #         # x_train = (x_train - mi) / (ma-mi)
    #         # mu = torch.mean(x_train)
    #         # std = torch.std(x_train)
    #         # x_train = (x_train - mu) / std

    #         y_train = y_train.reshape(486,).to(torch.int64)
    #         indices_train = torch.argsort(y_train)
    #         sorted_x_train = x_train[indices_train]
    #         sorted_y_train = y_train[indices_train]

    #         testdata_pooled = FedHeartDisease(train=False, pooled=True)
    #         test_data = torch.utils.data.DataLoader(testdata_pooled, batch_size=254, shuffle=False, num_workers=0)
    #         x_test, y_test  = next(iter(test_data)) 
    #         y_test = y_test.reshape(254,).to(torch.int64)
            
    #     else:
    #         sorted_x_train = []
    #         sorted_y_train = []
    #         x_list = []
    #         y_list = []
    #         client_set = [0, 1, 2, 3]
    #         for center in client_set:
    #             traindata = FedHeartDisease(train=True, pooled=False, center=center)
    #             traindata_size = len(traindata)
    #             train_data = torch.utils.data.DataLoader(traindata, batch_size=traindata_size, shuffle=False, num_workers=0)
    #             x_train, y_train  = next(iter(train_data))
    #             y_train = y_train.reshape(traindata_size,).to(torch.int64)
    #             sorted_x_train.append(x_train)
    #             sorted_y_train.append(y_train)

    #             testdata = FedHeartDisease(train=False, pooled=False, center=center)
    #             testdata_size = len(testdata)
    #             test_data = torch.utils.data.DataLoader(testdata, batch_size=testdata_size, shuffle=False, num_workers=0)
    #             x, y  = next(iter(test_data))
    #             x_list.append(x)
    #             y_list.append(y.reshape(testdata_size,).to(torch.int64))
    #         x_test = torch.cat(x_list, dim=0)
    #         y_test = torch.cat(y_list)

    #         labels_set = []
    #         # for cid in range(N):
    #         for cid in range(len(client_set)):
    #             # idx = [int(val) for val in client_set[cid]]
    #             labels_set.append(set(np.array(sorted_y_train[cid])))

    #             labels_count = [0]*2
    #             for label in np.array(sorted_y_train[cid]):
    #                 labels_count[int(label)] += 1
    #             print('cid: {}, number of labels: {}/2.'.format(cid, len(labels_set[cid])))
    #             print(labels_count)
    #         print()
            
            

    print('Using {} dataset!\n'.format(name))


    return sorted_x_train, sorted_y_train, x_test, y_test