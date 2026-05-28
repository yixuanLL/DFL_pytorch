# -*- coding: utf-8 -*-

import argparse
import copy
import importlib
import os

import torch

from models.client import Client
from models.server import Server
from utils.budgets_accountant import BudgetsAccountant
from utils.create_dataset import prepare_local_dataset
from utils.dataloader import loader_cifar10_lora_split
from utils.dpsgd_utils import privacy_check_gdp_dpdr, privacy_check_gdp_dpsgd
from utils.main_utils import print_accuracy_and_loss, setup_seed


MODEL_PARAMS = {
    'CIFAR10': (3 * 32 * 32, 10),
}


def _load_pretrained_if_needed(model, args, device):
    if not args.pretrained_path:
        return
    if args.model == 'cnn5_lora':
        mod = importlib.import_module('models.cnn5_lora')
        mod.load_pretrained_cnn5_lora(model, args.pretrained_path, map_location=device)
    else:
        checkpoint = torch.load(args.pretrained_path, map_location=device)
        state = checkpoint.get('model_state_dict', checkpoint)
        model.load_state_dict(state, strict=False)
        print('Loaded pretrained weights from %s' % args.pretrained_path)


def _save_checkpoint(model, args):
    if not args.save_checkpoint:
        return
    os.makedirs(os.path.dirname(args.save_checkpoint), exist_ok=True)
    torch.save({'model_state_dict': model.state_dict()}, args.save_checkpoint)
    print('Saved checkpoint to %s' % args.save_checkpoint)


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[if using gpu]", torch.cuda.is_available())
    setup_seed(args.seed)

    x_train, y_train, x_test, y_test = loader_cifar10_lora_split(
        split=args.lora_split,
        private_size=args.private_size,
        split_seed=args.split_seed,
    )
    dataset, args.num_clients = prepare_local_dataset(
        args.noniid, args.num_clients, y_train, args.seed, args.dataset)

    data_size = len(dataset[0])
    noise_multiplier_g = args.noise_multiplier_g
    noise_multiplier_p = args.noise_multiplier_p
    noise_multiplier_a = args.noise_multiplier_a
    budget_accountant = None

    if args.dp:
        budget_accountant = BudgetsAccountant(
            args.eps, args.delta, noise_multiplier_g, noise_multiplier_p, noise_multiplier_a)
        if args.DR:
            flag = privacy_check_gdp_dpdr(
                local_dataset_size=data_size,
                local_batch_size=args.batch_size,
                epochs=args.global_round * args.sample_ratio,
                epsilon_budget=args.eps,
                delta_budget=args.delta,
                sigma_perp=noise_multiplier_p,
                sigma_alpha=noise_multiplier_a,
                sigma_g=noise_multiplier_g,
                steps_dr=args.steps_dr,
                steps_interval=args.steps_interval,
            )
            noise_list = [noise_multiplier_p, noise_multiplier_a, noise_multiplier_g]
        else:
            flag = privacy_check_gdp_dpsgd(
                local_dataset_size=data_size,
                local_batch_size=args.batch_size,
                epochs=args.global_round * args.sample_ratio,
                epsilon_budget=args.eps,
                delta_budget=args.delta,
                sigma_g=noise_multiplier_g,
            )
            noise_list = [noise_multiplier_g]
        if not flag:
            print("noise multiplier is too small to satisfy privacy!", noise_list, args.eps)
            exit(0)

    clients = []
    for i in range(args.num_clients):
        clients.append(Client(
            x_train=x_train,
            y_train=y_train,
            x_test=x_test,
            y_test=y_test,
            dataset=dataset[i],
            dataname=args.dataset,
            batch_size=args.batch_size,
            FLalg=args.FLalg,
            dp=args.dp,
            DR=args.DR,
            DRV2=False,
            DRtest=False,
            Topk=False,
            cpl=False,
            kfilter=False,
            rate_dr=1,
            local_round=args.local_round,
            grad_norm=args.grad_norm,
            grad_perp_norm=args.grad_perp_norm,
            lr=args.lr,
            momentum=args.momentum,
            budget_accountant=budget_accountant,
            device=device,
            opt=args.opt,
            alg='none',
            num_clients=args.num_clients,
            clip_paral=args.clip_paral,
            steps_interval=args.steps_interval,
            steps_dr=args.steps_dr,
        ))

    print('client noise multiplier is %f, %f, %f' % (
        noise_multiplier_g, noise_multiplier_p, noise_multiplier_a))

    model_path = '%s.%s' % ('models', args.model)
    print('Model:', args.model)
    mod = importlib.import_module(model_path)
    model_cls = getattr(mod, 'Model')
    server = Server(
        num_clients=args.num_clients,
        sample_ratio=args.sample_ratio,
        model=model_cls,
        x_test=x_test,
        y_test=y_test,
        model_param=MODEL_PARAMS[args.dataset],
        device=device,
        grad_norm=args.grad_norm,
        perp_grad_norm=args.grad_perp_norm,
        clip_paral=args.clip_paral,
        budget_accountant=budget_accountant,
        glr=args.glr,
    )
    server.init_alg(dp=args.dp, FLalg=args.FLalg)
    global_model = server.init_global_model().to(device)
    _load_pretrained_if_needed(global_model, args, device)
    server.global_last_grad = [
        p.data.to(device) for p in global_model.parameters() if p.requires_grad
    ]

    communication_round = args.global_round // args.local_round
    print('the communication_round is %d' % communication_round)
    accuracy_accountant = []

    for r in range(communication_round):
        candidates = server.sample_clients([pin for pin in range(args.num_clients) if clients[pin].precheck()])
        last_parameters = [p for p in copy.deepcopy(global_model).parameters() if p.requires_grad]
        global_last_model = copy.deepcopy([weight.data.to(device) for weight in global_model.state_dict().values()])

        for participant in candidates:
            clients[participant].download(copy.deepcopy(global_model), server.global_last_grad)
            model_state, _, _, bytes2 = clients[participant].local_update()
            server.aggregate(model_state, global_last_model)

        global_model = server.update(global_last_model)
        current_trainable = [p for p in global_model.parameters() if p.requires_grad]
        server.global_last_grad = [
            (p1.data - p2.data).to(device) for p1, p2 in zip(current_trainable, last_parameters)
        ]

        test_accuracy, test_loss = server.test(global_model)
        accuracy_accountant.append(test_accuracy)
        print_accuracy_and_loss(r, test_accuracy, test_loss)
        if args.num_clients == 1 and bytes2[2]:
            accuracy_accountant = bytes2[2]

    _save_checkpoint(global_model, args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='CIFAR10')
    parser.add_argument('--model', type=str, default='cnn5_lora')
    parser.add_argument('--lora_split', type=str, default='private')
    parser.add_argument('--private_size', type=int, default=10000)
    parser.add_argument('--split_seed', type=int, default=0)
    parser.add_argument('--pretrained_path', type=str, default='')
    parser.add_argument('--save_checkpoint', type=str, default='')
    parser.add_argument('--FLalg', type=str, default='FedAvg')
    parser.add_argument('--DR', type=bool, default=False)
    parser.add_argument('--global_round', type=int, default=10)
    parser.add_argument('--local_round', type=int, default=10)
    parser.add_argument('--noniid', type=bool, default=False)
    parser.add_argument('--num_clients', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--dp', type=bool, default=True)
    parser.add_argument('--eps', type=float, default=3.0)
    parser.add_argument('--delta', type=float, default=1e-5)
    parser.add_argument('--noise_multiplier_g', type=float, default=0.786338)
    parser.add_argument('--noise_multiplier_p', type=float, default=0.792232)
    parser.add_argument('--noise_multiplier_a', type=float, default=6.457354)
    parser.add_argument('--grad_norm', type=float, default=0.2)
    parser.add_argument('--grad_perp_norm', type=float, default=0.2)
    parser.add_argument('--clip_paral', type=float, default=0.05)
    parser.add_argument('--sample_ratio', type=float, default=1)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--lr', type=float, default=0.5)
    parser.add_argument('--glr', type=float, default=1)
    parser.add_argument('--momentum', type=float, default=0.0)
    parser.add_argument('--opt', type=str, default='sgd')
    parser.add_argument('--steps_dr', type=int, default=200)
    parser.add_argument('--steps_interval', type=int, default=40000)
    args = parser.parse_args()

    parsed = vars(args)
    max_len = max([len(ii) for ii in parsed.keys()])
    fmt_string = '\t%' + str(max_len) + 's : %s'
    print('Arguments:')
    for key_pair in parsed.items():
        print(fmt_string % key_pair)
    main(args)
    print('[FINISHED]')
