import torch
from opt_einsum.contract import contract

class A():
    def __init__(self) -> None:
        self.last_grad = [torch.tensor([3.,4.,5.]), torch.tensor([3.,4.,5.])]
        self.grad_samples = [torch.tensor([[1.,1.,1.],[0.3,0.4,0.55]]), torch.tensor([[1.,1.,1.],[0.3,0.4,0.55]])]
        # norm = torch.stack([p.reshape(-1).norm(2, dim=-1) for p in self.last_grad], dim=0).norm(2, -1)
        norm = [p.reshape(-1).norm(2, dim=-1) for p in self.last_grad]
        self.rate_dr = 0.5
        self.last_grad = [p/n for p,n in zip(self.last_grad,norm)] 

    def decompose_grad(self):
        per_param_norms = [g.reshape(len(g), -1).norm(2, dim=-1) for g in self.grad_samples] # norm of per laryer of per sample gradient
        # print('per_param_norms', per_param_norms[0].numpy())
        last_grad_norms = [g.reshape(-1).norm(2, dim=-1) for g in self.last_grad] # norm of per laryer of last gradient
        costheta = [torch.sum(g.reshape(len(g), -1)*(lg.reshape(-1)), dim=1)/(g_norm*lg_norm) for (g, lg, g_norm, lg_norm) in zip(self.grad_samples, self.last_grad, per_param_norms, last_grad_norms)]
        gi_paral_test = [torch.reshape(gn*cos, [len(gn)]+[1]*len(lg.shape)) * torch.tile(lg.unsqueeze(0),[len(gn)]+[1]*len(lg.shape)) for gn, cos, lg in zip(per_param_norms, costheta, self.last_grad)]
        gi_paral = []
        for i in range(len(self.last_grad)):
            tmp = torch.zeros_like(self.grad_samples[i])
            for j in range(len(self.grad_samples[i])):
                tmp[j] = per_param_norms[i][j] * costheta[i][j] * self.last_grad[i]
            gi_paral.append(tmp)
        # print(gi_paral_test[0].numpy())
        # print(gi_paral[0].numpy())
        gi_perp = [(g-gl) for g, gl in zip(self.grad_samples, gi_paral)]
        # norm = torch.stack([p.reshape(len(p), -1).norm(2, dim=-1) for p in gi_perp], dim=1).norm(2, dim=-1) # norm of whole gi_perp
        norm = [p.reshape(len(p), -1).norm(2, dim=-1) for p in gi_perp] # norm of per laryer of per gi_perp
        # print('norm', norm)
        # print(gi_perp[0].numpy())
        gi_perp = [g/n.unsqueeze(dim=1) for g,n in zip(gi_perp, norm)]     

        return gi_perp, costheta, per_param_norms

    def recover_grad(self, gi_perp, g_norm, costheta):
        if self.last_grad == []:
            g = [torch.sum(gp, dim=0) for gp in gi_perp]
        else:
            # 为什么norm后有0.2的acc误差？
            g = [contract("i,i...", gn*torch.sqrt(1-cos**2), gp)  + contract("i,i...", gn * cos, torch.tile(lg.unsqueeze(0),[len(gn)]+[1]*len(lg.shape))) for gp, gn, cos, lg in zip(gi_perp, g_norm, costheta, self.last_grad)]
            # g = [contract("i,i...", gn/gn, gp) + contract("i,i...", gn * cos, torch.tile(lg.unsqueeze(0),[len(gn)]+[1]*len(lg.shape))) for gp, gn, cos, lg in zip(gi_perp, g_norm, costheta, self.last_grad)]
            for gp, gn, cos, lg in zip(gi_perp, g_norm, costheta, self.last_grad):
                tmp = gp[0]*0
                for i in range(len(cos)):
                    tmp = gn[i]*torch.sqrt(1-cos[i]**2) * gp[i] #gn[i]*cos[i] * lg + 
                    print('tmp_perp', tmp.numpy())
                    tmp = gn[i]*cos[i] * lg 
                    print('tmp_paral', tmp.numpy())
                
                g_perp = contract("i,i...", gn*torch.sqrt(1-cos**2), gp)
                print('g_perp', g_perp.numpy())
                g_paral = contract("i,i...", gn * cos, torch.tile(lg.unsqueeze(0),[len(gn)]+[1]*len(lg.shape)))
                print('g_paral', g_paral.numpy())
        per_param_sum = [torch.sum(g, dim=0) for g in self.grad_samples]
        return g

    def top_mask(self, gi_perp, mod='topk'):
        gi_perp_summed = [torch.sum(g, dim=0).reshape(-1) for g in gi_perp] # sum of a batch
        print('gi_perp', gi_perp[0].numpy())
        print('gi_perp_summed', gi_perp_summed[0].numpy())
        masked_g = []
        for i in range(len(gi_perp_summed)):
            g = gi_perp_summed[i]
            topk_num = int(g.shape[0]*self.rate_dr)
            # if g.shape[0]<20:
            #     topk_num = g.shape[0]
            if mod == 'topk':
                idx_topk = torch.topk(torch.abs(g), topk_num)[1]
            else:
                idx_topk = torch.randint(0, len(g), (topk_num,))
            print(idx_topk)
            #  for DP
            # idx_topk = exp_topk(idx_topk, topk_num, 100)
            mask = torch.zeros_like(g)
            mask[idx_topk] = 1
            for j in range(len(gi_perp[i])):
                gi_perp[i][j] *= torch.reshape(mask, gi_perp[i].shape[1:])
            return gi_perp

a = A()
print('last_grad', a.last_grad[0].numpy())
print('grad_sample', a.grad_samples[0].numpy())

gi_perp, costheta, per_param_norms = a.decompose_grad()
# print('gi_perp', gi_perp[0].numpy())
# print('cos', costheta[0].numpy())
# print('per_param_norm', per_param_norms[0].numpy())
# g = a.recover_grad(gi_perp, per_param_norms, costheta)
# print('reverse_grad', g[0])
gi_perp_topk = a.top_mask(gi_perp)
print('gi_perp_topk', gi_perp_topk[0].numpy())