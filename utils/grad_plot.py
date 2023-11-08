
import matplotlib.pyplot as plt
import torch

def grad_plot(log):
    global_round = len(log)
    clean = []
    clean27 = []
    noisy = []
    estimate = []
    rounds = 0
    for i in range(global_round):
        local_round = len(log[i])
        for j in range(local_round):
            c, n, e = log[i][j]
            clean.append(c[0].reshape(-1)[0])
            clean27.append(c[1].reshape(-1)[0])
            # noisy.append(n[0].reshape(-1)[0])
            # estimate.append(e[0].reshape(-1)[0])
            rounds += 1 

    plt.switch_backend('agg')
    r = range(rounds)
    plt.plot(r, clean27, 'skyblue', label='clean grad, dim27')
    plt.plot(r, clean, 'yellowgreen', label='clean grad, dim0')

    # plt.plot(r, noisy, 'g', label='noisy grad')
    # plt.plot(r, estimate, 'c', label='estimate grad')

    plt.ylabel('Gradients')
    plt.xlabel('Rounds')
    plt.legend(loc='lower right', fontsize=8)

    # plt.title('FLamby $\epsilon$=0.5', fontsize=9)
    plt.show()
    root_path = '/home/yliu270/workspace/DFL_pytorch/'
    plt.savefig(root_path+'grad.png', dpi=600)
    plt.close()

def grad_var_t(log): #variance of certain dimension along time step (var of T gradients)
    global_round = len(log)
    clean = []
    noisy = []
    estimate = []
    rounds = 0
    for i in range(global_round):
        local_round = len(log[i])
        for j in range(local_round):
            c, n, e = log[i][j] # parameters of a whole model
            clean.append(grad_flat(c))
            noisy.append(grad_flat(n))
            estimate.append(grad_flat(e))
            rounds += 1 

    clean = torch.var(torch.stack(clean, dim=0), dim=0)
    noisy =  torch.var(torch.stack(noisy, dim=0), dim=0)
    # estimate = torch.stack(estimate, dim=1)
    dim_num = len(clean)

    plt.switch_backend('agg')
    r = range(rounds)
    d = range(dim_num)
    plt.bar(d, clean, color='skyblue', label='clean grad', alpha=0.6)
    plt.bar(d, noisy, color='yellowgreen', label='noisy grad', alpha=0.6)
    # plt.hist(clean, bins=10, color='skyblue', alpha=0.6, label='clean grad')
    # plt.plot(r, estimate, 'c', label='estimate grad')

    plt.ylabel('Gradients Var OVer time')
    plt.xlabel('Dimension ID')
    plt.legend(loc='best', fontsize=8)

    # plt.title('FLamby $\epsilon$=0.5', fontsize=9)
    plt.show()
    root_path = '/home/yliu270/workspace/DFL_pytorch/'
    plt.savefig(root_path+'grad_var_t.png', dpi=600)
    plt.close()

def grad_flat(param):
    vec = torch.tensor([])
    for p in param:
        vec = torch.cat((vec, p.reshape(-1)))
    return vec

def grad_var(log): #variance of certain dimension along time step (var of t-1 gradients)
    global_round = len(log)
    clean = []
    noisy = []
    estimate = []
    rounds = 0
    for i in range(global_round):
        local_round = len(log[i])
        for j in range(local_round):
            c, n, e = log[i][j] # parameters of a whole model
            clean.append(grad_flat(c))
            noisy.append(grad_flat(n))
            estimate.append(grad_flat(e))
            rounds += 1 

    clean = torch.var(torch.stack(clean, dim=0), dim=0)
    noisy =  torch.var(torch.stack(noisy, dim=0), dim=0)
    estimate = torch.var(torch.stack(estimate, dim=0), dim=0)
    dim_num = len(clean)

    plt.switch_backend('agg')
    r = range(rounds)
    d = range(dim_num)

    plt.bar(d, noisy, color='yellowgreen', label='noisy grad', alpha=0.46)
    plt.bar(d, estimate, color='gold', label='estimate grad', alpha=0.6)
    plt.bar(d, clean, color='skyblue', label='clean grad', alpha=0.6)
    # plt.hist(clean, bins=10, color='skyblue', alpha=0.6, label='clean grad')


    plt.ylabel('Gradients Var OVer time')
    plt.xlabel('Dimension ID')
    plt.legend(loc='best', fontsize=8)

    # plt.title('FLamby $\epsilon$=0.5', fontsize=9)
    plt.show()
    root_path = '/home/yliu270/workspace/DFL_pytorch/'
    plt.savefig(root_path+'grad_var.png', dpi=600)
    plt.close()

def grad_var_t(log): #variance of certain dimension along time step
    global_round = len(log)
    clean = []
    noisy = []
    estimate = []
    rounds = 0
    # noise_var = (26.352314 * 0.1 / 4.0)**2
    noise_var =(11.785113* 0.1 / 4.0)**2
    for i in range(global_round):
        local_round = len(log[i])
        for j in range(local_round):
            c, n, e = log[i][j] # parameters of a whole model
            clean.append(grad_flat(c))
            noisy.append(grad_flat(n))
            estimate.append(grad_flat(e))
            rounds += 1 

    d = range(len(clean))
    r = range(rounds)

    c = torch.stack(clean, dim=0)
    n = torch.stack(noisy, dim=0)
    e = torch.stack(estimate, dim=0)
    clean = []
    noisy = []
    estimate = []
    for i in r:
        clean.append(torch.var(c[0:i+1], dim=0)) # [r, d]
        noisy.append(torch.var(n[0:i+1], dim=0))
        estimate.append(torch.var(e[0:i+1], dim=0))
    clean = torch.stack(clean, dim=0)
    noisy = torch.stack(noisy, dim=0)
    noise = noisy - clean
    clean_estimate = noisy - noise_var
    estimate = torch.stack(estimate, dim=0)

    plt.switch_backend('agg')

    plt.plot(r, noisy[:,27], color='yellowgreen', label='noisy grad dim27', marker='s')
    plt.plot(r, clean[:,27], color='skyblue', label='clean grad dim27')
    plt.plot(r, (noisy-noise_var)[:,27], color='gold', label='noisy-noise var dim27')
    plt.plot(r, (clean+noise_var)[:,27], color='silver', label='noise+noise var dim27')
    plt.plot(r, estimate[:,27], color='lightsalmon', label='estimate grad dim27')
    plt.plot(r, noise[:,27], color='violet', label='noise dim27')
    plt.ylabel('Gradients Var along Time')
    plt.xlabel('Epochs')
    plt.legend(loc='best', fontsize=8)

    # plt.title('FLamby $\epsilon$=0.5', fontsize=9)
    plt.show()
    root_path = '/home/yliu270/workspace/DFL_pytorch/'
    plt.savefig(root_path+'grad_var_t_dim27.png', dpi=600)
    plt.close()

    plt.plot(r, noisy[:,0], color='darkolivegreen', label='noisy grad dim0', alpha=0.6)
    plt.plot(r, clean[:,0], color='dodgerblue', label='clean grad dim0', alpha=0.6)
    plt.plot(r, clean_estimate[:,0], color='orange', label='noisy-clean var dim0', alpha=0.6)
    plt.plot(r, estimate[:,0], color='tomato', label='estimate grad dim0', alpha=0.6)



    # plt.hist(clean, bins=10, color='skyblue', alpha=0.6, label='clean grad')


    plt.ylabel('Gradients Var along Time')
    plt.xlabel('Epochs')
    plt.legend(loc='best', fontsize=8)

    # plt.title('FLamby $\epsilon$=0.5', fontsize=9)
    plt.show()
    root_path = '/home/yliu270/workspace/DFL_pytorch/'
    plt.savefig(root_path+'grad_var_t_dim0.png', dpi=600)
    plt.close()