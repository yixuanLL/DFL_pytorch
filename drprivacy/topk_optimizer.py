from opacus.optimizers import DPOptimizer
from opacus.optimizers.optimizer import _check_processed_flag, _generate_noise, _mark_as_processed
from torch.optim import Optimizer
from typing import Callable, List, Optional, Union
import torch
from opt_einsum.contract import contract
import copy
from utils.dpsgd_utils import exp_topk

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
        self.rate_dr = max_grad_norm[2]
        self.last_grad = []

        
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

        self.topk_process()
        
        self.add_noise()

        self.scale_grad()

        if self.step_hook:
            self.step_hook(self)

        self._is_last_step_skipped = False
        return True  

    def topk_process(self): # baseline V.S. dr_process
        grad = copy.deepcopy([p.grad_sample for p in self.params])
        gi_topk = self.top_mask(grad)        
        gi_topk_clipped = self.clip_g_perp(gi_topk)
        g_topk_clipped = [torch.sum(g, dim=0) for g in gi_topk_clipped]
        for p,gi in zip(self.params, g_topk_clipped):
            if p.summed_grad is not None:
                p.summed_grad += gi
            else:
                p.summed_grad = gi
        return

    def top_mask(self, gi_perp, mod='topk'):
        gi_perp_summed = [torch.sum(g, dim=0).reshape(-1) for g in gi_perp] # sum of a batch
        for i in range(len(gi_perp_summed)):
            g = gi_perp_summed[i]
            topk_num = int(g.shape[0]*self.rate_dr)
            # if g.shape[0]<20:
            #     topk_num = g.shape[0]
            if mod == 'topk':
                # idx_topk = torch.topk(torch.abs(g), topk_num)[1]
                idx_topk = exp_topk(torch.abs(g), topk_num, 1/(2*topk_num))
            else:
                idx_topk = torch.randint(0, len(g), (topk_num,))
            #  for DP
            # idx_topk = exp_topk(idx_topk, topk_num, 100)
            mask = torch.zeros_like(g)
            mask[idx_topk] = 1
            for j in range(len(gi_perp[i])):
                gi_perp[i][j] *= torch.reshape(mask, gi_perp[i].shape[1:])
        return gi_perp

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
            grad = contract("i,i...", per_sample_clip_factor, grad_sample)
            p.grad_sample = torch.reshape(per_sample_clip_factor, [len(grad_sample)]+[1]*(len(grad_sample.shape)-1)) * grad_sample

            # if p.summed_grad is not None:
            #     p.summed_grad += grad
            # else:
            #     p.summed_grad = grad

            _mark_as_processed(p.grad_sample)


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

            _mark_as_processed(p.summed_grad)

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








        
            
