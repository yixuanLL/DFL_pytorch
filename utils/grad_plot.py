
import matplotlib.pyplot as plt
import torch
import numpy as np
from opt_einsum.contract import contract
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA

def alpha_plot(log):
    global_round = len(log)
    alpha0 = []
    alpha1 = []
    rounds = 0
    for i in range(global_round):
        local_round = len(log[i])
        for j in range(local_round):
            # only print the certain batch of each local round
            # if j%(local_round/2) != 0:
            #     continue
            a = log[i][j]
            alpha0.append(a[0].cpu())
            alpha1.append(a[1].cpu())
            rounds += 1 
    plt.switch_backend('agg')
    r = range(rounds)
    plt.plot(r, alpha0, 'skyblue', label='a0')
    plt.plot(r, alpha1, 'green', label='a1')
    plt.ylabel('alpha')
    plt.xlabel('Rounds')
    plt.legend(loc='lower right', fontsize=8)

    plt.show()
    root_path = '/home/yliu270/workspace/DFL_pytorch/'
    plt.savefig(root_path+'alpha.png', dpi=600)
    plt.close()

def loss_plot(losses):
    global_round = len(losses)
    loss = []
    rounds = 0
    for i in range(global_round):
        local_round = len(losses[i])
        for j in range(local_round):
            loss.append(losses[i][j])
            rounds += 1
    plt.switch_backend('agg')
    r = range(rounds)
    plt.plot(r, loss, 'skyblue', label='loss')
    plt.ylabel('Loss')
    plt.xlabel('Rounds')
    plt.legend(loc='lower right', fontsize=8)

    plt.show()
    root_path = '/home/yliu270/workspace/DFL_pytorch/'
    plt.savefig(root_path+'loss.png', dpi=600)
    plt.close()

def grad_plot(log):
    global_round = len(log)
    norm = []
    clean = []
    clean1 = []
    noisy = []
    noisy1 = []
    estimate = []
    rounds = 0
    for i in range(global_round):
        local_round = len(log[i])
        for j in range(local_round):
            # only print the certain batch of each local round
            if j%(local_round/2) != 0:
                continue
            c, n, e = log[i][j]
            clean.append(c[0].reshape(-1)[0].cpu())
            clean1.append(c[1].reshape(-1)[0].cpu())
            norm.append(torch.norm(grad_flat(c), dim=0).cpu())
            noisy.append(n[0].reshape(-1)[0].cpu())
            noisy1.append(n[1].reshape(-1)[0].cpu())
            # estimate.append(e[0].reshape(-1)[0])
            rounds += 1 


    plt.switch_backend('agg')
    r = range(rounds)
    plt.plot(r, clean, 'skyblue', label='clean grad, layer1')
    plt.plot(r, clean1, 'yellowgreen', label='clean grad, layer2')

    plt.plot(r, noisy, 'g', label='noisy grad, layer1')
    plt.plot(r, noisy1, 'olive', label='noisy grad, layer2')
    # plt.plot(r, estimate, 'c', label='estimate grad')

    plt.ylabel('Gradients')
    plt.xlabel('Rounds')
    plt.legend(loc='lower right', fontsize=8)

    # plt.title('FLamby $\epsilon$=0.5', fontsize=9)
    plt.show()
    root_path = '/home/yliu270/workspace/DFL_pytorch/'
    plt.savefig(root_path+'grad.png', dpi=600)
    plt.close()

    plt.plot(r, norm, 'skyblue', label='grad norm')
    plt.ylabel('Gradients Norm')
    plt.xlabel('Rounds')
    plt.legend(loc='lower right', fontsize=8)

    plt.show()
    root_path = '/home/yliu270/workspace/DFL_pytorch/'
    plt.savefig(root_path+'norm.png', dpi=600)
    plt.close()

def grad_plot_t(log):
    global_round = len(log)
    clean = []
    clean1 = []
    clean_all = []
    clean1_all = []
    clean_window  = [0]
    clean1_window = [0]
    noisy = []
    estimate = []
    rounds = 0
    local_round = 0
    for i in range(global_round):
        local_round = len(log[i])
        for j in range(local_round):
            c, n, e = log[i][j]
            # only print the certain batch of each local round
            if j%(local_round/2) == 0:
                clean.append(c[0].reshape(-1)[0])
                clean1.append(c[1].reshape(-1)[0])
                rounds += 1 
                clean_all.append(c[0].reshape(-1)[0])
                clean1_all.append(c[1].reshape(-1)[0])
            # noisy.append(n[0].reshape(-1)[0])
            # estimate.append(e[0].reshape(-1)[0])
    T = 10

    for i in range(1, len(clean_all)):
        start = max(0, i-T)
        clean_window.append(torch.mean(torch.stack(clean_all[start:i], dim=0), dim=0))
        clean1_window.append(torch.mean(torch.stack(clean1_all[start:i], dim=0), dim=0))

    plt.switch_backend('agg')
    r = range(int(len(clean_all)/rounds)-1,len(clean_all), int(len(clean_all)/rounds))
    rs = range(len(clean_all))
    plt.plot(r, clean, 'skyblue', label='clean grad, layer1')
    plt.plot(r, clean1, 'yellowgreen', label='clean grad, layer2')
    plt.plot(rs, clean_window, 'blue', label='clean grad accumulative, layer1')
    plt.plot(rs, clean1_window, 'olive', label='clean grad accumulative, layer2')

    # plt.plot(r, noisy, 'g', label='noisy grad')
    # plt.plot(r, estimate, 'c', label='estimate grad')

    plt.ylabel('Gradients')
    plt.xlabel('Rounds')
    plt.legend(loc='lower right', fontsize=8)

    # plt.title('FLamby $\epsilon$=0.5', fontsize=9)
    plt.show()
    root_path = '/home/yliu270/workspace/DFL_pytorch/'
    plt.savefig(root_path+'grad_accum.png', dpi=600)
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
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = 'cpu'
    vec = torch.tensor([]).to(device)
    for p in param:
        vec = torch.cat((vec, p.reshape(-1).to(device)))
    return vec

def grad_flat_per_sample(param):
    device = 'cpu'
    vec = torch.tensor([]).to(device)
    ns = len(param[0])
    for p in param:
        vec = torch.cat((vec, p.reshape(ns, -1).to(device)), 1)
    return vec

def grad_var(log): #variance of certain dimension along time step (var of t-1 gradients)
    plt.switch_backend('agg')
    global_round = len(log)
    clean = []
    noisy = []
    estimate = []
    rounds = 0
    clean_l = [[] for i in range(len(log[0][0][0]))]
    color= ['r', 'g', 'b', 'y', 'tomato', 'yellowgreen', 'silver', 'violet']
    for i in range(global_round):
        local_round = len(log[i])
        for j in range(local_round):
            c, n, e = log[i][j] # parameters of a whole model
            # clean.append(grad_flat(c))
            # noisy.append(grad_flat(n))
            # estimate.append(grad_flat(e))
            rounds += 1 
            for l in range(len(c)):
                # print(c[l].reshape(-1))
                clean_l[l].append(c[l].reshape(-1))
    print(clean_l)
    for k in range(len(clean_l)):
        cl = torch.var(torch.stack(clean_l[k], dim=0), dim=0)
        dim_num = len(cl)
        plt.plot(range(dim_num), cl, color=color[k], label=str(k))

    # clean = torch.var(torch.stack(clean, dim=0), dim=0)
    # noisy =  torch.var(torch.stack(noisy, dim=0), dim=0)
    # estimate = torch.var(torch.stack(estimate, dim=0), dim=0)
    # dim_num = len(clean)

    # plt.switch_backend('agg')
    # r = range(rounds)
    # d = range(dim_num)

    # plt.bar(d, noisy, color='yellowgreen', label='noisy grad', alpha=0.46)
    # plt.bar(d, estimate, color='gold', label='estimate grad', alpha=0.6)
    # plt.bar(d, clean, color='skyblue', label='clean grad')
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
    noise_var = (26.352314 * 0.1 / 4.0)**2
    # noise_var =(11.785113* 0.1 / 4.0)**2
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
    plt.plot(r, (clean+noise_var)[:,27], color='silver', label='clean+noise var dim27')
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

    plt.plot(r, noisy[:,0], color='yellowgreen', label='noisy grad dim0', marker='s')
    plt.plot(r, clean[:,0], color='skyblue', label='clean grad dim0')
    plt.plot(r, (noisy-noise_var)[:,0], color='gold', label='noisy-noise var dim0')
    plt.plot(r, (clean+noise_var)[:,0], color='silver', label='clean+noise var dim0')
    plt.plot(r, estimate[:,0], color='lightsalmon', label='estimate grad dim0')
    plt.plot(r, noise[:, 0], color='violet', label='real noise dim0')


    # plt.hist(clean, bins=10, color='skyblue', alpha=0.6, label='clean grad')


    plt.ylabel('Gradients Var along Time')
    plt.xlabel('Epochs')
    plt.legend(loc='best', fontsize=8)

    # plt.title('FLamby $\epsilon$=0.5', fontsize=9)
    plt.show()
    root_path = '/home/yliu270/workspace/DFL_pytorch/'
    plt.savefig(root_path+'grad_var_t_dim0.png', dpi=600)
    plt.close()

def grad_dist(log):
    global_round = len(log)
    grads = []
    grads_p = []
    grads_d = []
    ratio = []
    rounds = 0
    for i in range(global_round):
        local_round = len(log[i])
        for r in range(local_round):
            j = local_round - r -1
            a = log[i][j]
            grad = (grad_flat(a[0])).numpy()
            grad_p = (grad_flat(a[1])).numpy()
            grad_d = (grad_flat(a[2])).numpy()

            rounds += 1 
            if j % 5 == 0:
                # print(grad_d)
                # print(grad)
                plt.switch_backend('agg')
                plt.hist(grad, bins=5, label='grad', alpha=1)
                # plt.hist(grad_p,  bins=500, color='green', label='grad_perp', alpha=0.4)
                # plt.hist(grad_d,  bins=500, color='red', label='grad_diff', alpha=0.2)
        plt.ylabel('Amounts')
        plt.xlabel('Value')
        # plt.xlim(-0.01,0.01)
        plt.legend(loc='lower right', fontsize=8)

        plt.show()
        root_path = '/local/scratch/yliu270/workspace/DFL_pytorch/pics/res_grad/'
        plt.savefig(root_path+str(rounds-1)+' epochs.png', dpi=600)
        plt.close()
        #     grads.append(np.linalg.norm(grad))
        #     grads_p.append(np.linalg.norm(grad_p))
        #     ratio.append(np.linalg.norm(grad_p)/np.linalg.norm(grad))
        #     grads_d.append(np.linalg.norm(grad_d))
        # grads_d[0] = 0
        # grads_p[0] = 0
        # ratio[0] = 0
        # plt.switch_backend('agg')
        # r = range(rounds)
        # plt.plot(r, grads, color='skyblue', label='grad', alpha=0.8)
        # plt.plot(r, grads_p,  color='green', label='grad_perp', alpha=0.6)
        # plt.plot(r, ratio,  color='red', label='ratio', alpha=0.6)
        # plt.plot(r, grads_d,  color='orange', label='grad_diff', alpha=0.6)
        # plt.ylabel('Norm')
        # plt.xlabel('Steps')
        # plt.legend(loc='lower right', fontsize=8)

        # plt.show()
        # root_path = '/local/scratch/yliu270/workspace/DFL_pytorch/pics/res_grad/'
        # plt.savefig(root_path+'Norm_drv3.png', dpi=600)
        # plt.close()

def decompose_grad(grad_samples, last_grad):
    # last_grad = [torch.mean(g, dim=0) for g in last_grad]
    last_grad_norms = [g.reshape(-1).norm(2, dim=-1) for g in last_grad] # norm of per laryer of last gradient
    paral_alpha = [torch.sum(g.reshape(-1)*lg.reshape(-1))/(lg_norm*lg_norm) for (g, lg, lg_norm) in zip(grad_samples, last_grad, last_grad_norms)]
    gi_paral = [paral * lg for paral, lg in zip(paral_alpha, last_grad)]
    gi_perp = [(g-gl) for g, gl in zip(grad_samples, gi_paral)] 
    return gi_perp, paral_alpha

def clip_g_perp(g_perp):
    per_param_norms = [g.reshape(-1).norm(2, dim=-1) for g in g_perp] # norm of per laryer of per sample gradient
    per_sample_norms = torch.stack(per_param_norms).norm(2) # norm of per sample gradient
    per_sample_clip_factor = (0.1 / (per_sample_norms + 1e-6)).clamp(max=1.0) # clip [ max min ]
    g_perp_clipped = [per_sample_clip_factor * g for g in g_perp]
    return g_perp_clipped

def clip_alpha(alpha):
    per_param_norms = [g.norm(2, dim=-1) for g in alpha] # norm of per laryer of per sample gradient
    per_sample_norms = torch.stack(per_param_norms).norm(2) # norm of per sample gradient
    per_sample_clip_factor = (0.1 / (per_sample_norms + 1e-6)).clamp(max=1.0) # clip [ max min ]
    g_perp_clipped = [per_sample_clip_factor * g for g in alpha]
    return g_perp_clipped

def recover_grad(g_perp_noisy, alpha_noisy, last_grad):
    g_noisy = [gp + a * lg for gp, a, lg in zip(g_perp_noisy, alpha_noisy, last_grad)]
    return g_noisy

def grad_2D(log):

    global_round = len(log)
    gpn = [0]
    bn=[0]
    gn=[]
    cn=[]
    g1 = []
    g2 = []
    c1 = []
    c2 = []
    gp1 = []
    gp2 = []
    rounds = 0
    for i in range(global_round):
        local_round = len(log[i])
        for j in range(local_round):
            # only print the certain batch of each local round
            # if j%(local_round/2) != 0:
            #     continue
            g, c, _ = log[i][j]
            g1.append(g[0].reshape(-1)[0].cpu())
            g2.append(g[1].reshape(-1)[0].cpu())
            c1.append(c[0].reshape(-1)[0].cpu())
            c2.append(c[1].reshape(-1)[0].cpu())
            print(g[1])
            if j>0:
                gl, _, _ = log[i][j-1]
                gi_perp, alpha = decompose_grad(g, gl)
                gp = clip_g_perp(gi_perp) 
                g_noisy = recover_grad(gp, alpha, gl)
                bias = [gn-gi for gn, gi in zip(g_noisy, g)]

                # gp = [torch.mean(g, dim=0) for g in gi_perp]
                gp1.append(gp[0].reshape(-1)[0].cpu())
                gp2.append(gp[1].reshape(-1)[0].cpu())
                gpn.append(torch.norm(grad_flat(gp), dim=0).cpu())
                bn.append(torch.norm(grad_flat(bias), dim=0).cpu())

            gn.append(torch.norm(grad_flat(g), dim=0).cpu())
            cn.append(torch.norm(grad_flat(c), dim=0).cpu())
            
                
            rounds += 1 

            if j%40 == 0:
                plt.switch_backend('agg')
                r = range(rounds)
                plt.plot(g1, g2, 'o', label='grad')
                plt.plot(c1, c2, '.', label='diff')
                plt.plot(gp1, gp2, '.', label='perp')

                plt.ylabel('layer2 dim1')
                plt.xlabel('layer1 dim1')
                plt.legend(loc='lower right', fontsize=8)

                # plt.title('FLamby $\epsilon$=0.5', fontsize=9)
                plt.show()
                root_path = '/local/scratch/yliu270/workspace/DFL_pytorch/pics/'
                title= 'epoch'+ str(j)
                plt.savefig(root_path+title, dpi=600)
                plt.close()
                g1 = []
                g2 = []
                c1 = []
                c2 = []
                gp1=[]
                gp2=[]

        plt.switch_backend('agg')
        r = range(rounds)
        # plt.plot(r, gn, '-', label='grad')
        # plt.plot(r, cn, 'r', label='diff')
        # plt.plot(r, gpn,'g', label='perp')
        plt.plot(r, bn,'g', label='bias')

        plt.ylabel('norm')
        plt.xlabel('epochs')
        plt.legend(loc='lower right', fontsize=8)

        # plt.title('FLamby $\epsilon$=0.5', fontsize=9)
        plt.show()
        root_path = '/local/scratch/yliu270/workspace/DFL_pytorch/pics/'
        title= "Norm"
        plt.savefig(root_path+title, dpi=600)
        plt.close()


def grad_norm_save(log, dataset, model):
    global_round = len(log)
    grads = []
    grads_p = []
    grads_d = []
    ratio = []
    lgrads = []
    lgrads_p = []
    lgrads_d = []
    lratio = []
    rounds = 0
    root_path = '/local/scratch/yliu270/workspace/DFL_pytorch/pics/res_grad/'
    file_path = root_path+dataset + '_' + model + '.txt'
    f = open(file_path, 'w') 

    for i in range(global_round):
        local_round = len(log[i])
        for j in range(local_round):
            # if dataset == 'SVHN':
            if True:
                a = log[i][j]
                gn = torch.tensor(a[0]).cpu().data.numpy()
                gpn = torch.tensor(a[1]).cpu().data.numpy()
                gdn = torch.tensor(a[2]).cpu().data.numpy()
            else:
                a = log[i][j]
                g = a[0]
                grad = (grad_flat(a[0]))
                gn = torch.norm(grad, dim=0).cpu().numpy()
                if i==0 and j==0:
                    gpn = 0
                    gdn = 0
                else:
                    if i!=0 and j==0:
                        gl = log[i-1][-1][0]
                    else:
                        gl = log[i][j-1][0]
                    gi_perp, alpha = decompose_grad(g, gl)
                    gi_diff = [gi-gli for gi, gli in zip(g, gl)]
                    # gp = clip_g_perp(gi_perp) 
                    # alpha = clip_alpha(alpha)
                    # g_noisy = recover_grad(gp, alpha, gl)
                    # bias = [gn-gi for gn, gi in zip(g_noisy, g)]

                    gpn = torch.norm(grad_flat(gi_perp), dim=0).cpu().numpy()
                    gdn = torch.norm(grad_flat(gi_diff), dim=0).cpu().numpy()
            grads.append(str(gn))
            grads_p.append(str(gpn))
            grads_d.append(str(gdn))
            ratio.append(str(gpn/gn))
            lgrads.append(gn)
            lgrads_p.append(gpn)
            lgrads_d.append(gdn)
            lratio.append(gpn/gn)
            rounds += 1
        str_j = str(j)
        f.write(str_j + ','+'grad_norm,'+','.join(grads) + '\n')
        f.write(str_j + ','+'gperp_norm,'+','.join(grads_p) + '\n')
        f.write(str_j + ','+'gdiff_norm,'+','.join(grads_d) + '\n')
        f.write(str_j + ','+'gpn2gn,'+','.join(ratio) + '\n')

        grads = []
        grads_p = []
        grads_d = []
        ratio = []
    plt.switch_backend('agg')
    r = range(rounds)
    plt.plot(r, lgrads, color='skyblue', label='grad', alpha=0.8)
    plt.plot(r, lgrads_p,  color='green', label='grad_perp', alpha=0.6)
    plt.plot(r, lratio,  color='red', label='ratio', alpha=0.6)
    plt.plot(r, lgrads_d,  color='gold', label='grad_diff', alpha=0.6)
    plt.ylabel('Norm')
    plt.xlabel('Steps')
    plt.legend(loc='lower right', fontsize=8)

    plt.show()
    root_path = '/local/scratch/yliu270/workspace/DFL_pytorch/pics/res_grad/'
    plt.savefig(root_path+'Norm_dp.png', dpi=600)
    plt.close()
    f.close()


def grad_pca(log, dataset):

    global_round = len(log)
    rounds = 0
    pca = PCA()
    pipe = Pipeline([('scaler', StandardScaler()),
                 ('pca', pca)])
    for i in range(global_round):
        local_round = len(log[i])
        for j in range(local_round):
            # only print the certain batch of each local round
            # if j%(local_round/2) != 0:
            #     continue
            g, c, p = log[i][j]
            if c == 0:
                continue
            ns = len(g[0])
            # 按向量 pca
            # g1 = grad_flat(g).numpy().reshape(ns,-1)
            # c1 = grad_flat(c).numpy().reshape(ns,-1)
            # gp1 = grad_flat(p).numpy().reshape(ns,-1)
            # for m in range(len(g)): #按层
                # g1=g[m].cpu().numpy().reshape(ns,-1)
                # c1=c[m].cpu().numpy().reshape(ns,-1)
                # gp1=p[m].cpu().numpy().reshape(ns,-1)
            #按层和维度
            rounds += 1 
            for m in range(len(g)-1):
                if True:
                    gtx = g[m].cpu().numpy().reshape(ns,-1)[:,0]
                    gty = g[m+1].cpu().numpy().reshape(ns,-1)[:,0]
                    ctx = c[m].cpu().numpy().reshape(ns,-1)[:,0]
                    cty = c[m+1].cpu().numpy().reshape(ns,-1)[:,0]            
                    gptx = p[m].cpu().numpy().reshape(ns,-1)[:,0]
                    gpty = p[m+1].cpu().numpy().reshape(ns,-1)[:,0]
                    
                    plt.switch_backend('agg')

                    r = range(rounds)
                    plt.scatter(gtx, gty, label='grad', marker='o')
                    plt.scatter(ctx, cty, label='diff', marker='+')
                    plt.scatter(gptx, gpty, label='perp', marker='*', c="none", edgecolors='g')
                # if j%49 == 1:
                # if True: #按向量pca
                    # gt = pipe.fit_transform(np.array(g1))
                    # ct = pipe.fit_transform(np.array(c1))
                    # gpt = pipe.fit_transform(np.array(gp1))

                    
                    # plt.switch_backend('agg')

                    # r = range(rounds)
                    # plt.scatter(gt[:,0], gt[:,1], label='grad', marker='o')
                    # plt.scatter(ct[:,0], ct[:,1], label='diff', marker='+')
                    # plt.scatter(gpt[:,0], gpt[:,1], label='perp', marker='*', c="none", edgecolors='g')

                    # plt.ylabel('layer2 dim1')
                    # plt.xlabel('layer1 dim1')
                    # plt.xlim(-20,20)
                    # plt.ylim(-20,20)
                    plt.legend(loc='lower right', fontsize=8)

                    # plt.title('FLamby $\epsilon$=0.5', fontsize=9)
                    plt.show()
                    root_path = '/local/scratch/yliu270/workspace/DFL_pytorch/pics/res_grad/'
                    title= 'pca_'+dataset+ str(j) + 'layer' + str(m) + '.pdf'
                    plt.savefig(root_path+title, dpi=600)
                    plt.close()

def grad_pca(log, dataset):

    global_round = len(log)
    rounds = 0
    pca = PCA()
    pipe = Pipeline([('scaler', StandardScaler()),
                 ('pca', pca)])
    for i in range(global_round):
        local_round = len(log[i])
        for j in range(local_round):
            # only print the certain batch of each local round
            # if j%(local_round/2) != 0:
            #     continue
            g, c, p = log[i][j]
            if c == 0:
                continue
            ns = len(g[0])
            # 按向量 pca
            # g1 = grad_flat(g).numpy().reshape(ns,-1)
            # c1 = grad_flat(c).numpy().reshape(ns,-1)
            # gp1 = grad_flat(p).numpy().reshape(ns,-1)
            # for m in range(len(g)): #按层
                # g1=g[m].cpu().numpy().reshape(ns,-1)
                # c1=c[m].cpu().numpy().reshape(ns,-1)
                # gp1=p[m].cpu().numpy().reshape(ns,-1)
            #按层和维度
            rounds += 1 
            for m in range(len(g)-1):
                if True:
                    gtx = g[m].cpu().numpy().reshape(ns,-1)[:,0]
                    gty = g[m+1].cpu().numpy().reshape(ns,-1)[:,0]
                    ctx = c[m].cpu().numpy().reshape(ns,-1)[:,0]
                    cty = c[m+1].cpu().numpy().reshape(ns,-1)[:,0]            
                    gptx = p[m].cpu().numpy().reshape(ns,-1)[:,0]
                    gpty = p[m+1].cpu().numpy().reshape(ns,-1)[:,0]
                    
                    plt.switch_backend('agg')

                    r = range(rounds)
                    plt.scatter(gtx, gty, label='grad', marker='o')
                    plt.scatter(ctx, cty, label='diff', marker='+')
                    plt.scatter(gptx, gpty, label='perp', marker='*', c="none", edgecolors='g')
                # if j%49 == 1:
                # if True: #按向量pca
                    # gt = pipe.fit_transform(np.array(g1))
                    # ct = pipe.fit_transform(np.array(c1))
                    # gpt = pipe.fit_transform(np.array(gp1))

                    
                    # plt.switch_backend('agg')

                    # r = range(rounds)
                    # plt.scatter(gt[:,0], gt[:,1], label='grad', marker='o')
                    # plt.scatter(ct[:,0], ct[:,1], label='diff', marker='+')
                    # plt.scatter(gpt[:,0], gpt[:,1], label='perp', marker='*', c="none", edgecolors='g')

                    # plt.ylabel('layer2 dim1')
                    # plt.xlabel('layer1 dim1')
                    # plt.xlim(-20,20)
                    # plt.ylim(-20,20)
                    plt.legend(loc='lower right', fontsize=8)

                    # plt.title('FLamby $\epsilon$=0.5', fontsize=9)
                    plt.show()
                    root_path = '/local/scratch/yliu270/workspace/DFL_pytorch/pics/res_grad/'
                    title= 'pca_'+dataset+ str(j) + 'layer' + str(m) + '.pdf'
                    plt.savefig(root_path+title, dpi=600)
                    plt.close()
def grad_pca_vec(log, dataset):

    global_round = len(log)
    rounds = 0
    pca = PCA()
    pipe = Pipeline([('scaler', StandardScaler()),
                 ('pca', pca)])
    for i in range(global_round):
        local_round = len(log[i])
        for j in range(local_round):
            # only print the certain batch of each local round
            # if j%(local_round/2) != 0:
            #     continue
            g, c, p = log[i][j]
            if c == 0:
                continue
            ns = len(g[0])
            # 按向量 pca
            g1 = grad_flat_per_sample(g).numpy()
            c1 = grad_flat_per_sample(c).numpy()
            gp1 = grad_flat_per_sample(p).numpy()
            #按层和维度
            rounds += 1 

            if True: #按向量pca
                gt = pipe.fit_transform(np.array(g1))
                ct = pipe.fit_transform(np.array(c1))
                gpt = pipe.fit_transform(np.array(gp1))

                
                plt.switch_backend('agg')

                r = range(rounds)
                plt.scatter(gt[:,0], gt[:,1], label='grad', marker='o')
                plt.scatter(ct[:,0], ct[:,1], label='diff', marker='+')
                plt.scatter(gpt[:,0], gpt[:,1], label='perp', marker='*', c="none", edgecolors='g')

                plt.ylabel('layer2 dim1')
                plt.xlabel('layer1 dim1')
                # plt.xlim(-20,20)
                # plt.ylim(-20,20)
                plt.legend(loc='lower right', fontsize=8)

                # plt.title('FLamby $\epsilon$=0.5', fontsize=9)
                plt.show()
                root_path = '/local/scratch/yliu270/workspace/DFL_pytorch/pics/res_grad/'
                title= 'pcavec_'+dataset+ str(j) + '.pdf'
                plt.savefig(root_path+title, dpi=600)
                plt.close()
def grad_norm_hist(log, dataset):
    # CIFAR10 norm=500 lr=0.005 batch size=256
    cl = ['orange', 'yellowgreen',  'dodgerblue', 'slategrey','blueviolet', 'darkcyan']
    a = [1, 0.8, 0.6,0.4]
    global_round = len(log)
    rounds = 0
    # plt.figure(41)
    # fignum = 410
    fig, ax = plt.subplots(4,1, sharey=True, sharex=True)
    plt.switch_backend('agg')
    steps = 0
    for i in range(global_round):
        local_round = len(log[i])
        for j in range(local_round):
            gn, c, p = log[i][j]
            gn = gn.cpu().numpy()
            # 按向量
            # g1 = grad_flat_per_sample(g).numpy()
            # gn = np.linalg.norm(g1, axis=1)

            rounds += 1 
            # if j%49 == 1:
            if True:


                if j >= 4:
                    break
                ax[j].hist(gn,  bins=5,  color='cornflowerblue', label='Steps '+str(steps), alpha=1)
                steps += 50
                # legend1 = ax.legend([line1], ["line1"], loc="upper right")
                # ax.add_artist(legend1)
                ax[j].legend(loc='upper right', fontsize=8)


        plt.xlabel('Norm')
        plt.ylabel('Frequency')
        plt.xlim(2,8)
        # plt.ylim(-20,20)
        plt.xticks(fontsize=6)
        plt.yticks(fontsize=6)
        # plt.legend(loc='upper right', fontsize=8)

        # plt.title('FLamby $\epsilon$=0.5', fontsize=9)
        plt.show()
        root_path = '/local/scratch/yliu270/workspace/DFL_pytorch/pics/res_grad/'
        title= 'grad_hist_'+ dataset+ str(j) + '.pdf'
        plt.savefig(root_path+title, dpi=600)
        plt.close()