# -*- coding: utf-8 -*-
# @Author : yixuan
# @File : main.py


import copy
import torch
import argparse
import importlib
from utils.create_dataset import prepare_local_dataset
from utils.dataloader import loader
from models.client_diff2 import Client
print('__client_diff2__')
# from models.client import Client
# print('__client__')
from models.server import Server
from utils.dpsgd_utils import compute_noise_multiplier
from utils.budgets_accountant import BudgetsAccountant
from utils.main_utils import save_progress, print_accuracy_and_loss, setup_seed
import os
from utils.grad_plot import grad_plot, grad_var, grad_var_t, loss_plot, grad_plot_t, alpha_plot,grad_dist
# os.environ['CUDA_VISIBLE_DEVICES'] ='2'

MODEL_PARAMS={
    'MNIST': (784,10),
    'CIFAR10': (3*32*32,10),
    'CIFAR100': (3,100),
    'SVHN': (3,10),
    'FLamby': (13,2),
    'CAHouse': (8,1)
}
DATA_MODEL={
    'MNIST': "cnn",
    'CIFAR10': "cnn5",
    'CIFAR100': "resnet",
    'SVHN': "resnet",
    'FLamby': "mclr",
    'CAHouse': "mclr"
}
from torchvision import datasets, transforms
def main(args):
    accuracy_accountant = []
    privacy_accountant = []
    accum_nbytes_list1 = []
    accum_nbytes_list2 = []
    accum_nbytes1 = 0
    accum_nbytes2 = 0
    max_accum_budget_accountant = 0
    save_address = ''
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[if using gpu]", torch.cuda.is_available())

    # set seed
    setup_seed(args.seed)
    # prepare local dataset
    x_train, y_train, x_test, y_test = loader(args.dataset, args.noniid)
    dataset, args.num_clients = prepare_local_dataset(args.noniid, args.num_clients, y_train, args.seed, args.dataset)

    # set noise multiplier
    budget_accountant = None
    noise_multiplier = 0
    noise_multiplier_2 = 0

    
    # set clients
    clients = []
    for i in range(args.num_clients):
        if args.dp:
            eps = args.eps
            eps_2 = 10e6
            if args.DR or args.DRtest or args.DRV2:
                eps_2 = args.eps_2
                # eps = args.eps - eps_2
            try:
                data_size = len(dataset[i])
            except:
                data_size = len(y_train[i])
            noise_multiplier = compute_noise_multiplier(local_dataset_size=data_size,  local_batch_size=args.batch_size, T=args.global_round * args.sample_ratio,
                                                epsilon=eps, delta=args.delta)
            noise_multiplier_2 = compute_noise_multiplier(local_dataset_size=data_size, local_batch_size=args.batch_size, T=args.global_round * args.sample_ratio,
                                    epsilon=eps_2, delta=args.delta)
            noise_multiplier_3 = compute_noise_multiplier(local_dataset_size=data_size, local_batch_size=args.batch_size, T=args.global_round * args.sample_ratio,
                                    epsilon=eps_2+eps, delta=args.delta)
            budget_accountant = BudgetsAccountant(args.eps, args.delta, noise_multiplier, noise_multiplier_2, noise_multiplier_3)
                    
        clients.append(Client(x_train=x_train,
                        y_train=y_train,
                        x_test=x_test,
                        y_test=y_test,
                        dataset=dataset[i],
                        dataname=args.dataset,
                        batch_size=args.batch_size,
                        FLalg=args.FLalg, 
                        dp=args.dp,
                        DR=args.DR,
                        DRV2=args.DRV2,
                        DRtest=args.DRtest,
                        Topk=args.Topk,
                        cpl=args.cpl,
                        kfilter=args.kf,
                        rate_dr=args.rate_dr,
                        local_round=args.local_round,
                        grad_norm=args.grad_norm,
                        grad_perp_norm=args.grad_perp_norm,
                        lr=args.lr,
                        momentum=args.momentum,
                        budget_accountant=budget_accountant,
                        device=device,
                        opt=args.opt,
                        num_clients=args.num_clients,
                        clip_paral=args.clip_paral))
    print('client noise multiplier is %f, %f, %f' % (noise_multiplier, noise_multiplier_2, noise_multiplier_3)) 
    
    # set server
    model_path = '%s.%s' % ('models', DATA_MODEL[args.dataset])
    print('Model:', DATA_MODEL[args.dataset])
    mod = importlib.import_module(model_path)
    model = getattr(mod, 'Model')
    server = Server(num_clients=args.num_clients, sample_ratio=args.sample_ratio, model=model, x_test=x_test, y_test=y_test, model_param=MODEL_PARAMS[args.dataset], device=device, grad_norm=args.grad_norm, perp_grad_norm=args.grad_perp_norm, clip_paral=args.clip_paral, budget_accountant=budget_accountant, glr=args.glr)    
    server.init_alg(dp=args.dp, FLalg=args.FLalg) # init server algo: fedavg + dp
    global_model = server.init_global_model() # global model
    server.global_last_grad = [p.data.to(device) for p in global_model.parameters()]

    
    # communication round
    communication_round = args.global_round // args.local_round
    print('the communication_round is %d' % communication_round)
    log = []
    loss = []
    # start communication
    for r in range(communication_round):   
        # precheck and pick up candidates
        candidates = server.sample_clients([pin for pin in range(args.num_clients) if clients[pin].precheck()]) 
        last_parameters = copy.deepcopy(global_model).parameters()
        global_last_model = copy.deepcopy([weight.data.to(device) for weight in global_model.state_dict().values()])
        # local update
        for p_id, participant in enumerate(candidates):
            # download global model
            clients[participant].download(copy.deepcopy(global_model), server.global_last_grad)
            
            # update
            model_state, accum_budget_accountant, bytes1, bytes2 = clients[participant].local_update()
            
            # communication cost
            accum_nbytes1 += bytes1 / (1024 * 1024)
            accum_nbytes2 = 0
            # accum_nbytes2 += bytes2 / (1024 * 1024)
            if accum_budget_accountant:
                max_accum_budget_accountant = max(max_accum_budget_accountant, accum_budget_accountant)
            # aggregate
            server.aggregate(model_state, global_last_model)
            
            # log
            # if p_id == 0:
            #     log.append(bytes2[0])
            #     loss.append(bytes2[1])
            
            # if args.dp:
            #     print('for client: %d and delta: %.5f the budget: %.8f and the cost budget: %.8f \n'
            #           % ((participant+1), args.delta, clients[participant].budget_accountant.epsilon, clients[participant].budget_accountant.accum_bgts))
        # load average weight
        global_model = server.update(global_last_model)
        
        # for global_last_grad
        server.global_last_grad = [(p1.data-p2.data).to(device) for p1,p2 in zip(global_model.parameters(), last_parameters)]
        # server.global_last_grad = [(p1.data-p2.data) for p1,p2 in zip(global_model.parameters(), last_parameters)]
        
        # test
        test_accuracy, test_loss = server.test(global_model)
        accuracy_accountant.append(test_accuracy)
        print_accuracy_and_loss(r, test_accuracy, test_loss)
        
        if args.dp:
            privacy_accountant.append(max_accum_budget_accountant)
            if args.DR:
                accum_nbytes_list1.append(accum_nbytes1)
                accum_nbytes_list2.append(accum_nbytes2)
                save_address = save_progress(args, accuracy_accountant, privacy_accountant, accum_nbytes_list1, accum_nbytes_list2)
            else:
               save_address = save_progress(args, accuracy_accountant, privacy_accountant) 
        else:
            save_address = save_progress(args, accuracy_accountant)
        
        # if r > 3:
        #     break
    print(save_address)
    # grad_plot(log)
    # grad_plot_t(log)
    # loss_plot(loss)
    # grad_var(log)
    # grad_var_t(log)
    # alpha_plot(log)
    # grad_dist(log)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_dir', type=str, default='result')
    parser.add_argument('--dataset', type=str, default='MNIST')
    parser.add_argument('--FLalg', type=str, default='FedAvg', help='Algorithm of FL')
    parser.add_argument('--DR', type=bool, default=False)
    parser.add_argument('--DRV2', type=bool, default=True)
    parser.add_argument('--DRtest', type=bool, default=False)
    parser.add_argument('--global_round', type=int, default=20)
    parser.add_argument('--local_round', type=int, default=20)
    parser.add_argument('--noniid', type=bool, default=False, help='if True, use noniid data')
    parser.add_argument('--num_clients', type=int, default=1) 
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--dp', type=bool, default=True, help='if True, use differential privacy')
    parser.add_argument('--eps', type=float, default=3)
    parser.add_argument('--eps_2', type=float, default=0.02)
    parser.add_argument('--delta', type=float, default=1e-5, help='differential privacy parameter')
    parser.add_argument('--grad_norm', type=float, default=1)
    parser.add_argument('--grad_perp_norm', type=float, default=0.1)
    parser.add_argument('--sample_ratio', type=float, default=1)
    parser.add_argument('--seed', type=int, default=0)
    # parser.add_argument('--model', type=str, default='cnn5')
    parser.add_argument('--lr', type=float, default=4)
    parser.add_argument('--glr', type=float, default=1, help='global learning rate')
    parser.add_argument('--momentum', type=float, default=0.)
    parser.add_argument('--Topk', type=bool, default=False)
    parser.add_argument('--cpl', type=bool, default=False)
    parser.add_argument('--kf', type=bool, default=False)
    parser.add_argument('--opt', type=str, default='sgd')
    parser.add_argument('--rate_dr', type=float, default=1, help='sparse rate in directional reduction')
    parser.add_argument('--clip_paral', type=float, default=10, help='parallel alpha bound')
    args = parser.parse_args() 

    # print arguments
    try: parsed = vars(parser.parse_args())
    except IOError as msg: parser.error(str(msg))
    maxLen = max([len(ii) for ii in parsed.keys()]);
    fmtString = '\t%' + str(maxLen) + 's : %s';
    print('Arguments:')
    for keyPair in parsed.items(): print(fmtString % keyPair)
    
    main(args)
    print('[FINISHED]')