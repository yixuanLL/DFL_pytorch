from opacus.optimizers import DPOptimizer
from opacus.optimizers.optimizer import _check_processed_flag, _generate_noise, _mark_as_processed
from torch.optim import Optimizer
from typing import Callable, List, Optional, Union
import torch
from opt_einsum.contract import contract
import copy
from utils.dpsgd_utils import exp_topk
import math
import numpy as np
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# add noise during decompose, and set norm as instant
class TopkDPOptimizer(DPOptimizer):
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
        # print('===Topk Dr DP test===')
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
        self.rate = max_grad_norm[3]
        self.last_grad = []
        self.last_normratio = []
        self.norm = 1
        self.global_last_grad = []

        
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
        self.scale_grad()


        if self.step_hook:
            self.step_hook(self)

        self._is_last_step_skipped = False
        return True  

    def dr_process(self):
        if self.last_grad == []:
            self.cold_start()
            return
        gi_topk, last_grad_topk, mask = self.top_mask(self.last_grad, 'topk')
        # last_grad_topk = self.last_grad
        # gi_topk = self.grad_samples
        gi_perp_topk, paral_alpha_topk = self.decompose_grad(last_grad_topk, gi_topk)   
        g_perp_topk = self.clip_g_perp(gi_perp_topk)  
        g_perp_topk_clean = copy.deepcopy(g_perp_topk)
        self.add_noise_sum(g_perp_topk, self.noise_multiplier, self.perp_grad_norm)


        # preserve paral factor
        if self.last_grad != []:
            # clip_p = 0.05 # mnist
            clip_p = 0.001
            # print(paral_alpha_topk)
            paral_alpha_topk = self.clip(paral_alpha_topk, clip_p)
            paral_alpha_topk_clean = copy.deepcopy(paral_alpha_topk)
            paral_alpha_topk = self.add_noise_mean(paral_alpha_topk, self.noise_multiplier_2, clip_p) 
        else:
            paral_alpha_topk = 0
        self.recover_grad(g_perp_topk, paral_alpha_topk, mask, g_perp_topk_clean, paral_alpha_topk_clean) 

    def decompose_grad(self, last_grad_topk, gi_topk):
        # if last_grad_topk == []:      
        #     return gi_topk, [torch.tensor(1.).to('cuda')]*8
        per_param_norms = [g.reshape(len(g), -1).norm(2, dim=-1) for g in gi_topk] # norm of per laryer of per sample gradient
        last_grad_norms = [g.reshape(-1).norm(2, dim=-1) for g in last_grad_topk] # norm of per laryer of last gradient
        costheta = [torch.mean(torch.sum(g.reshape(len(g), -1)*(lg.reshape(-1)), dim=1)/(g_norm*lg_norm+10e-6)) for (g, lg, g_norm, lg_norm) in zip(gi_topk, last_grad_topk, per_param_norms, last_grad_norms)]
        costheta = [torch.clamp(c, -1, 1) for c in costheta]
        gi_paral = [torch.reshape(gn*cos, [len(gn)]+[1]*len(lg.shape)) * torch.tile(lg.unsqueeze(0),[len(gn)]+[1]*len(lg.shape)) for gn, cos, lg in zip(per_param_norms, costheta, last_grad_topk)]
        gi_perp = [(g-gl) for g, gl in zip(gi_topk, gi_paral)] 
        paral_alpha =  [torch.mean(gn*cos, dim=0) for cos, gn in zip(per_param_norms, costheta)]
        return gi_perp, paral_alpha 

    def recover_grad(self, g_perp_topk, paral_alpha_topk, mask, g_perp_topk_clean, paral_alpha_topk_clean):
        # g_noisy = [ gp   + gn * lg * len(self.grad_samples[0]) for gp, gn, lg in zip(g_perp_topk, paral_alpha_topk, self.last_grad)]
        # g = [gp   + gn * lg * len(self.grad_samples[0]) for gp, gn, lg in zip(g_perp_topk, paral_alpha_topk, self.last_grad)]
        last_grad_topk = []
        last_grad_resi = []
        for i in range(len(self.last_grad)):
            last_grad_topk.append(self.last_grad[i] * mask[i])
            last_grad_resi.append(self.last_grad[i] - last_grad_topk[i])
        # g_noisy = [gp + (gn * lgk + lgr) * len(self.grad_samples[0]) for gp, gn, lgk, lgr in zip(g_perp_topk, paral_alpha_topk, last_grad_topk, last_grad_resi)]
        g_noisy = [gp + (gn * lgk) * len(self.grad_samples[0]) for gp, gn, lgk, lgr in zip(g_perp_topk, paral_alpha_topk, last_grad_topk, last_grad_resi)]
        g_clean = [gp + (gn * lgk + lgr) * len(self.grad_samples[0]) for gp, gn, lgk, lgr in zip(g_perp_topk_clean, paral_alpha_topk_clean, last_grad_topk, last_grad_resi)]
        for p,gi in zip(self.params, g_noisy):
            if p.summed_grad is not None:
                p.summed_grad += gi
            else:
                p.summed_grad = gi
        self.last_grad = [gi/torch.norm(gi, keepdim=False) for gi in g_noisy] # noisy last grad
        # self.last_grad = [gi/torch.norm(gi, keepdim=False) for gi in g_clean] # clean last grad
        return


    def clip_g_perp(self, g_perp):
        per_param_norms = [
            g.reshape(len(g), -1).norm(2, dim=-1) for g in g_perp
        ] # norm of per laryer of per sample gradient
        per_sample_norms = torch.stack(per_param_norms, dim=1).norm(2, dim=1) # norm of per sample gradient
        # print(per_sample_norms)
        per_sample_clip_factor = (
            self.perp_grad_norm / (per_sample_norms + 1e-6)
        ).clamp(max=1.0) # clip [ max min ]

        g_perp_clipped = []
        for p in g_perp:
            grad = contract("i,i...", per_sample_clip_factor, p) # mutiply [128] * [128, 16, 1, 8, 8] -> [16, 1, 8, 8] clip & sum
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
            # print(per_sample_norms)
            per_sample_clip_factor = (self.max_grad_norm / (per_sample_norms + 1e-6)).clamp(max=1.0)

        for p in self.params:
            _check_processed_flag(p.grad_sample)
            grad_sample = self._get_flat_grad_sample(p)
            # grad = contract("i,i...", per_sample_clip_factor, grad_sample)
            p.grad_sample = torch.reshape(per_sample_clip_factor, [len(grad_sample)]+[1]*(len(grad_sample.shape)-1)) * grad_sample

            _mark_as_processed(p.grad_sample)

    def clip(self, vec, clip_bound):
        # print(vec)
        norm = torch.stack(vec).norm(2, dim=0)
        clip_factor = (
            clip_bound / (norm + 1e-6)
        ).clamp(max=1.0)
        vec = [clip_factor * v for v in vec]
        return vec

    def top_mask(self, last_grad, mod='topk'):
        gi = copy.deepcopy([p.grad_sample for p in self.params])
        lg = copy.deepcopy([l.reshape(-1) for l in last_grad])
        mask = [0]*len(lg)
        for i in range(len(lg)): #each layer
            l = lg[i]
            length = len(l.reshape(-1))
            topk_num = int(length*self.rate)
            # if length<20:
            #     topk_num = length
            if mod == 'topk':
                idx_topk = (torch.topk(torch.abs(l), topk_num)[1])
            else:
                idx_topk = torch.randint(0, len(l), (topk_num,))
            maski = torch.zeros_like(l)
            maski[idx_topk] = 1
            mask[i] = torch.reshape(maski, gi[i].shape[1:])
            for j in range(len(gi[i])):
                gi[i][j] *= mask[i] #torch.reshape(maski, gi[i].shape[1:])

            lg[i] = self.last_grad[i] * mask[i]
        return gi, lg, mask

    def add_noise(self):
        """
        Adds noise to clipped gradients. Stores clipped and noised result in ``p.grad``
        """
        for p in self.params:
            _check_processed_flag(p.summed_grad)
            p.grad = (p.summed_grad).view_as(p)
            # print('noise/grad perp norm norm:{}'.format(torch.norm(noise) , torch.norm(p.summed_grad)))

            _mark_as_processed(p.summed_grad)

    def add_noise_mean(self, vec, noise_multiplier, sensitivity): 
        """
        Adds noise to clipped gradients. Stores clipped and noised result in ``p.grad``
        """
        std = noise_multiplier * sensitivity
        std /= (len(self.grad_samples[0]))
        # std = 10e-3
        # for i in range(len(tmp)):
            # a=0
            # noise = np.random.normal(loc=0, scale=std, size=(1,))
            # vec[i] += noise
        for v in vec:
            noise = torch.normal(
            mean=0,
            std=std,
            size=v.shape,
            device=device,
            generator=None,
        )
            v += noise
        return vec

    def add_noise_sum(self, vec, noise_multiplier, sensitivity):
        """
        Adds noise to clipped gradients. Stores clipped and noised result in ``p.grad``
        """
        std = noise_multiplier * sensitivity
        for v in vec:
            noise = torch.normal(
            mean=0,
            std=std,
            size=v.shape,
            device=device,
            generator=None,
        )
            v += noise
        return vec
    
    def cold_start(self):
        # sum all gi and add noises
        for p in self.params:
            grad_sample = self._get_flat_grad_sample(p)
            p.summed_grad = torch.sum(grad_sample, dim=0)
            noise = torch.normal(
            mean=0,
            std=self.max_grad_norm * self.noise_multiplier,
            size=p.summed_grad.shape,
            device=device,
            generator=None,
            )
            p.summed_grad = (p.summed_grad + noise).view_as(p)
        return




        
            
