from opacus.optimizers import DPOptimizer
from opacus.optimizers.optimizer import _check_processed_flag, _generate_noise, _mark_as_processed
from torch.optim import Optimizer
from typing import Callable, List, Optional, Union
import torch
from opt_einsum.contract import contract
import copy
from utils.dpsgd_utils import exp_topk
import math


# add noise during decompose, and set norm as instant
class DrDPOptimizerDiff2(DPOptimizer):
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
        # print('__DRDPtest__')
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
        self.last_normratio = []
        self.gt2 = []
        self.norm = 1
        self.global_last_grad = []
        self.g_perp_sum = []
        self.cos_sum = []
        self.log = []
        self.steps = 1
        self.num_sample = 128
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
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
        self.log = [[torch.mean(g, dim=0) for g in self.grad_samples], self.last_grad, []]

        if self.step_hook:
            self.step_hook(self)

        self._is_last_step_skipped = False
        return True  

    def dr_process(self):
        self.num_sample = len(self.grad_samples[0])
        gi_delta = self.diff()   
        g_delta = self.clip_and_sum(gi_delta, self.perp_grad_norm, 'g_perp') 
        g_delta_clean = copy.deepcopy(g_delta) 
        self.add_noise_sum(g_delta, self.noise_multiplier, self.perp_grad_norm)

        self.recover_grad(g_delta) 

    
    def diff(self):
        # if self.last_grad == []:
        if self.gt2 == []:
            return self.grad_samples
        d_i = [p-g for p,g in zip(self.gt2, self.grad_samples)]
        return d_i

    def recover_grad(self, g_delta):
        if self.last_grad == []:
            self.last_grad = g_delta
            g_noisy = g_delta #这里的浅复制会/sample size吗
        else:
            g_noisy = [d + g*len(self.last_grad[0]) for d, g in zip(g_delta, self.last_grad)] #sum of gradients
        for p,gi in zip(self.params, g_noisy):
            if p.summed_grad is not None:
                p.summed_grad += gi
            else:
                p.summed_grad = gi
        self.last_grad = g_noisy # mean of g_boisy


    def clip_and_sum(self, g_perp, clip_bound, mod):
        if mod == 'g_perp':
            per_param_norms = [g.reshape(len(g), -1).norm(2, dim=-1) for g in g_perp] # norm of per laryer of per sample gradient
            per_sample_norms = torch.stack(per_param_norms, dim=1).norm(2, dim=1) # norm of per sample gradient
        else:
            per_sample_norms = torch.stack(g_perp, dim=1).norm(2, dim=1) # norm of per sample gradient
        per_sample_clip_factor = (clip_bound / (per_sample_norms + 1e-6)).clamp(max=1.0) # clip [ max min ]

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

    def add_noise(self):
        """
        Adds noise to clipped gradients. Stores clipped and noised result in ``p.grad``
        """
        for p in self.params:
            _check_processed_flag(p.summed_grad)
            p.grad = (p.summed_grad).view_as(p)
            # print('noise/grad perp norm norm:{}'.format(torch.norm(noise) , torch.norm(p.summed_grad)))

            _mark_as_processed(p.summed_grad)

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
            device=self.device,
            generator=None,
        )
            v += noise
        return vec
