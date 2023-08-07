# -*- coding: utf-8 -*-
# @Author : yixuan
# @File : main.py


import copy
import torch
import argparse
import importlib
from utils.create_dataset import prepare_local_dataset
from utils.dataloader import loader
from models.client import Client
from models.server import Server
from utils.dpsgd_utils import compute_noise_multiplier
from utils.budgets_accountant import BudgetsAccountant
from utils.main_utils import save_progress, print_accuracy_and_loss, setup_seed

def main(args):
    accuracy_accountant = []
    privacy_accountant = []
    accum_nbytes_list1 = []
    accum_nbytes_list2 = []
    accum_nbytes1 = 0
    accum_nbytes2 = 0
    max_accum_budget_accountant = 0
    save_address = ''
    # set seed
    setup_seed(args.seed)
    # prepare local dataset
    x_train, y_train, x_test, y_test = loader(args.dataset)
    dataset = prepare_local_dataset(args.noniid, args.num_clients, y_train)

    # set noise multiplier
    budget_accountant = None
    noise_multiplier = 0
    if args.dp:
        noise_multiplier = compute_noise_multiplier(local_dataset_size=len(dataset[0]),
                                                    local_batch_size=args.batch_size,
                                                    T=args.global_round * args.sample_ratio,
                                                    epsilon=args.eps,
                                                    delta=args.delta)
        print('client noise multiplier is %f' % (noise_multiplier))

           
    # set clients
    clients = []
    for i in range(args.num_clients):
        if args.dp:
            budget_accountant = BudgetsAccountant(args.eps, args.delta, noise_multiplier)
        clients.append( Client(x_train=x_train,
                        y_train=y_train,
                        dataset=dataset[i],
                        batch_size=args.batch_size,
                        dp=args.dp,
                        DR=args.DR,
                        Topk=args.Topk,
                        rate_dr=args.rate_dr,
                        local_round=args.local_round,
                        grad_norm=args.grad_norm,
                        grad_perp_norm=args.grad_perp_norm,
                        budget_accountant=budget_accountant))

    # set server
    model_path = '%s.%s' % ('models', args.model)
    mod = importlib.import_module(model_path)
    model = getattr(mod, 'Model')
    server = Server(num_clients=args.num_clients, sample_ratio=args.sample_ratio, model=model, x_test=x_test, y_test=y_test)
    server.init_alg(dp=args.dp) # init server algo: fedavg + dp
    server_model = server.init_global_model() # global model

    # communication round
    communication_round = args.global_round // args.local_round
    print('the communication_round is %d' % communication_round)

    # start communication
    for r in range(communication_round):   
        # precheck and pick up candidates
        candidates = server.sample_clients([pin for pin in range(args.num_clients) if clients[pin].precheck()]) 

        # local update
        for p_id, participant in enumerate(candidates):
            # download global model
            clients[participant].download(copy.deepcopy(server_model))
            # update
            model_state, accum_budget_accountant, bytes1, bytes2 = clients[participant].local_update()
            # communication cost
            accum_nbytes1 += bytes1 / (1024 * 1024)
            accum_nbytes2 += bytes2 / (1024 * 1024)
            if accum_budget_accountant:
                max_accum_budget_accountant = max(max_accum_budget_accountant, accum_budget_accountant)
            # aggregate
            server.aggregate(model_state)
            
            if args.dp:
                print('for client: %d and delta: %.5f the budget: %.8f and the cost budget: %.8f \n'
                      % ((participant+1), args.delta, clients[participant].budget_accountant.epsilon, clients[participant].budget_accountant.accum_bgts))
        # load average weight
        global_model = server.update()

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
    print(save_address)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_dir', type=str, default='result')
    parser.add_argument('--dataset', type=str, default='MNIST')
    parser.add_argument('--DR', type=bool, default=False)
    parser.add_argument('--Topk', type=bool, default=False)
    parser.add_argument('--rate_dr', type=float, default=0.001, help='sparse rate in directional reduction')
    parser.add_argument('--global_round', type=int, default=10)
    parser.add_argument('--local_round', type=int, default=2)
    parser.add_argument('--noniid', type=bool, default=False, help='if True, use noniid data')
    parser.add_argument('--num_clients', type=int, default=2) 
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--dp', type=bool, default=True, help='if True, use differential privacy')
    parser.add_argument('--eps', type=float, default=2)
    parser.add_argument('--delta', type=float, default=1e-5, help='differential privacy parameter')
    parser.add_argument('--grad_norm', type=float, default=1)
    # parser.add_argument('--grad_perp_norm', type=float, default=0.8)
    parser.add_argument('--grad_perp_norm', type=float, default=1)
    parser.add_argument('--sample_ratio', type=float, default=1.0)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--model', type=str, default='cnn')
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