from opacus.optimizers import DPOptimizer
from opacus.optimizers.optimizer import _check_processed_flag, _generate_noise, _mark_as_processed
from torch.optim import Optimizer
from typing import Callable, List, Optional, Union
import torch
from opt_einsum.contract import contract
import copy
from utils.dpsgd_utils import exp_topk
import math

class DrDPOptimizertest(DPOptimizer):
    ## use max_grad_norm as grad norm, perp norm and rate_dr
    def __init__(self,
        optimizer: DPOptimizer,
        *,
        noise_multiplier: float,
        max_grad_norm: Optional[float],
        expected_batch_size: Optional[int],
        loss_reduction: str = "mean",
        generator=None,
        secure_mode: bool = False):
        # super(DPOptimizer, self).__init__(optimizer, noise_multiplier, max_grad_norm, expected_batch_size, loss_reduction, generator, secure_mode)
        # print('===Dr DP test===')
        self.original_optimizer = optimizer
        self.noise_multiplier = noise_multiplier
        self.loss_reduction = loss_reduction
        self.expected_batch_size = expected_batch_size
        self.step_hook = None
        self.generator = generator
        self.secure_mode = secure_mode

        self.param_groups = self.original_optimizer.param_groups
        self.defaults = self.original_optimizer.defaults
        self.state = self.original_optimizer.state
        self._step_skip_queue = []
        self._is_last_step_skipped = False

        for p in self.params:
            p.summed_grad = None
        
        self.max_grad_norm = max_grad_norm[0]
        self.perp_grad_norm = max_grad_norm[1]
        self.noise_multiplier_2 = max_grad_norm[2]
        self.last_grad = []
        self.global_last_grad = []
        self.g_perp_sum = []
        self.cos_sum = []

        
    def pre_step(
        self, closure: Optional[Callable[[], float]] = None
    ) -> Optional[float]:
        """
        Perform actions specific to ``DPOptimizer`` before calling
        underlying  ``optimizer.step()``

        Args:
            closure: A closure that reevaluates the model and
                returns the loss. Optional for most optimizers.
        """

        self.clip_and_accumulate()
        if self._check_skip_next_step():
            self._is_last_step_skipped = True
            return False

        self.dr_process()
        self.add_noise()
        
        # v1: use local last grad
        # norm = [p.grad.reshape(-1).norm(2, dim=-1) for p in self.params]
        # self.last_grad = [p.grad/n for p,n in zip(self.params, norm)] 

        # v2: use global last grad
        # if self.last_grad!=[]:
        # norm = [p.reshape(-1).norm(2, dim=-1) for p in self.last_grad]
        # self.last_grad = [p/n for p,n in zip(self.last_grad, norm)] 

        self.scale_grad()


        if self.step_hook:
            self.step_hook(self)

        self._is_last_step_skipped = False
        return True  

    def dr_process(self):
        gi_perp, costheta, per_param_norms, n = self.decompose_grad()        
        # gi_perp_topk = self.top_mask(gi_perp, 'rank')
        ############# TODO clip affect accuracy #################
        gi_perp_clipped = self.clip_g_perp(gi_perp)
        # preserve costheta
        # costheta = [torch.clamp(c, -0.1, 0.1) for c in costheta]
        costheta = self.add_noise_mean(costheta, self.noise_multiplier_2, 0.2) #noise=21.333 for eps=0.1; 42.66 for eps=0.05 | 100 rounds 114.48 for eps=0.05
        # costheta = [torch.clamp(c, -0.1, 0.1) for c in costheta]
        # costheta = self.add_noise_mean(costheta, 0.1, 1e-6, 0.2) # mean of cos
        # preserve norm
        sum_param_norms = [torch.sum(n) for n in per_param_norms]
        mean_param_norms = self.add_noise_mean([p/len(self.grad_samples[0]) for p in sum_param_norms], self.noise_multiplier_2, self.max_grad_norm)  #noise=21.333 for eps=0.1; 42.66 for eps=0.05
        # mean_param_norms = self.clip_noisy_norm(mean_param_norms)
        # mean_param_norms = self.add_noise_mean([p/len(self.grad_samples[0]) for p in sum_param_norms], 0.1, 1e-6, 1) 
        # mean_param_norms = [p/len(self.grad_samples[0]) for p in sum_param_norms]
        g_perp = self.recover_grad(gi_perp_clipped, mean_param_norms, costheta) 
        a = g_perp[0]*0
        # if self.last_grad != []:
        #     for i in range(8):
        #         a += gi_perp[0][i]*n[0][i]
        #     b=a-g_perp[0]
        # if self.g_perp_sum == []:
        #     self.g_perp_sum = g_perp
        #     self.cos_sum = costheta
        # else:
        #     self.g_perp_sum = [gps+gp for gps, gp in zip(self.g_perp_sum, g_perp)]
        #     self.cos_sum = [cs+c for cs, c in zip(self.cos_sum, costheta)]
 

    def decompose_grad(self):
        if self.last_grad == []:      
            return self.grad_samples, torch.tensor([1.]*len(self.grad_samples)).to('cuda'), torch.tensor([1.]*len(self.grad_samples)).to('cuda'), [0]
        per_param_norms = [g.reshape(len(g), -1).norm(2, dim=-1) for g in self.grad_samples] # norm of per laryer of per sample gradient
        last_grad_norms = [g.reshape(-1).norm(2, dim=-1) for g in self.last_grad] # norm of per laryer of last gradient
        costheta = [torch.mean(torch.sum(g.reshape(len(g), -1)*(lg.reshape(-1)), dim=1)/(g_norm*lg_norm)) for (g, lg, g_norm, lg_norm) in zip(self.grad_samples, self.last_grad, per_param_norms, last_grad_norms)]
        gi_paral = [torch.reshape(gn*cos, [len(gn)]+[1]*len(lg.shape)) * torch.tile(lg.unsqueeze(0),[len(gn)]+[1]*len(lg.shape)) for gn, cos, lg in zip(per_param_norms, costheta, self.last_grad)]
        gi_perp = [(g-gl) for g, gl in zip(self.grad_samples, gi_paral)]
        norm = [p.reshape(len(p), -1).norm(2, dim=-1) for p in gi_perp] # norm of per laryer of per gi_perp
        # ratio = [ngp/ng for ng, ngp in zip(per_param_norms, norm)]
        # print('ratio', ratio)
        gi_perp = [g/torch.reshape(n, [len(g)]+[1]*(len(g.shape)-1)) for g,n in zip(gi_perp,norm)]       
        return gi_perp, costheta, per_param_norms, norm

    def recover_grad(self, gi_perp, g_norm, costheta):
        if self.last_grad == []:
            g = [torch.sum(gp, dim=0) for gp in gi_perp]
            # g = [torch.sum(g, dim=0) for g in self.grad_samples]
            g_perp = g
        else:
            g_perp = [torch.sum(gn*torch.sqrt(1-cos**2) * gp, dim=0) for gp, gn, cos in zip(gi_perp, g_norm, costheta)]
            g = [torch.sum(gn*torch.sqrt(1-cos**2) * gp, dim=0)  + torch.sum(gn * cos * torch.tile(lg.unsqueeze(0),[len(gp)]+[1]*len(lg.shape)), dim=0) for gp, gn, cos, lg in zip(gi_perp, g_norm, costheta, self.last_grad)]
        # per_param_sum = [torch.sum(g, dim=0) for g in self.grad_samples]
        for p,gi in zip(self.params, g):
            if p.summed_grad is not None:
                p.summed_grad += gi
            else:
                p.summed_grad = gi
        return g_perp

    # def top_mask(self, gi_perp, mod='topk'):
    #     gi_perp_summed = [torch.sum(g, dim=0).reshape(-1) for g in gi_perp] # sum of a batch
    #     for i in range(len(gi_perp_summed)):
    #         g = gi_perp_summed[i]
    #         topk_num = int(g.shape[0]*self.rate_dr)
    #         # if g.shape[0]<20:
    #         #     topk_num = g.shape[0]
    #         if mod == 'topk':
    #             idx_topk = exp_topk(torch.abs(g), topk_num, 0.01/(2*topk_num))
    #         else:
    #             idx_topk = torch.randint(0, len(g), (topk_num,))
    #         mask = torch.zeros_like(g)
    #         mask[idx_topk] = 1
    #         # masked_g.append(gi_perp[i] * torch.reshape(mask, gi_perp[i].shape[1:]))
    #         for j in range(len(gi_perp[i])):
    #             gi_perp[i][j] *= torch.reshape(mask, gi_perp[i].shape[1:])
    #     return gi_perp

    #     # return masked_g

    def clip_g_perp(self, g_perp):
        per_param_norms = [
            g.reshape(len(g), -1).norm(2, dim=-1) for g in g_perp
        ] # norm of per laryer of per sample gradient
        per_sample_norms = torch.stack(per_param_norms, dim=1).norm(2, dim=1) # norm of per sample gradient
        per_sample_clip_factor = (
            self.perp_grad_norm / (per_sample_norms + 1e-6)
        ).clamp(max=1.0) # clip [ max min ]

        g_perp_clipped = []
        for p in g_perp:
            # grad_sample = self._get_flat_grad_sample(p) # change in to one tensor
            # grad = contract("i,i...", per_sample_clip_factor, p) # mutiply [128] * [128, 16, 1, 8, 8] -> [16, 1, 8, 8] clip & sum
            grad = torch.reshape(per_sample_clip_factor, [len(p)]+[1]*(len(p.shape)-1)) * p
            g_perp_clipped.append(grad)
        return g_perp_clipped


    def clip_and_accumulate(self):
        """
        Performs gradient clipping.
        Stores clipped and aggregated gradients into `p.summed_grad```
        """

        if len(self.grad_samples[0]) == 0:
            # Empty batch
            per_sample_clip_factor = torch.zeros((0,))
        else:
            per_param_norms = [
                g.reshape(len(g), -1).norm(2, dim=-1) for g in self.grad_samples
            ]
            per_sample_norms = torch.stack(per_param_norms, dim=1).norm(2, dim=1)
            per_sample_clip_factor = (
                self.max_grad_norm / (per_sample_norms + 1e-6)
            ).clamp(max=1.0)

        for p in self.params:
            _check_processed_flag(p.grad_sample)
            grad_sample = self._get_flat_grad_sample(p)
            # grad = contract("i,i...", per_sample_clip_factor, grad_sample)
            p.grad_sample = torch.reshape(per_sample_clip_factor, [len(grad_sample)]+[1]*(len(grad_sample.shape)-1)) * grad_sample

            # if p.summed_grad is not None:
            #     p.summed_grad += grad
            # else:
            #     p.summed_grad = grad

            _mark_as_processed(p.grad_sample)

    def clip_noisy_norm(self, mean_params_norm):
        norm = torch.stack(mean_params_norm).norm(2, dim=0)
        clip_factor = (
            self.max_grad_norm / (norm + 1e-6)
        ).clamp(max=1.0)
        mean_params_norm = [clip_factor * n for n in mean_params_norm]
        return mean_params_norm

    def add_noise(self):
        """
        Adds noise to clipped gradients. Stores clipped and noised result in ``p.grad``
        """
        for p in self.params:
            _check_processed_flag(p.summed_grad)

            noise = _generate_noise(
                std=self.noise_multiplier * self.perp_grad_norm,
                reference=p.summed_grad,
                generator=self.generator,
                secure_mode=self.secure_mode,
            )
            p.grad = (p.summed_grad + noise).view_as(p)
            # test without DP 
            # p.grad = (p.summed_grad).view_as(p)
            # print('noise/grad perp norm norm:{}'.format(torch.norm(noise) , torch.norm(p.summed_grad)))

            _mark_as_processed(p.summed_grad)

    def add_noise_mean(self, cos, noise_multiplier, sensitivity): #eps, delta, sensitivity):
        """
        Adds noise to clipped gradients. Stores clipped and noised result in ``p.grad``
        """
        # std = (sensitivity/eps) * math.sqrt(2 * math.log(1.25/delta))
        std = noise_multiplier * sensitivity
        std /= (len(self.grad_samples[0]))#**2
        for c in cos:
            noise = torch.normal(
            mean=0,
            std=std,
            size=c.shape,
            device='cuda',
            generator=None,
        )
            c += noise
        return cos







        
            
