from opacus.optimizers import DPOptimizer
from opacus.optimizers.optimizer import _check_processed_flag, _generate_noise, _mark_as_processed
from torch.optim import Optimizer
from typing import Callable, List, Optional, Union
import torch
from opt_einsum.contract import contract
import copy
from utils.dpsgd_utils import exp_topk
import math
from utils.kalman_filter import KalmanFilter

class KFilterOptimizer(DPOptimizer):
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

        self.max_grad_norm = max_grad_norm[0]
        self.perp_grad_norm = max_grad_norm[1]
        self.rate_dr = max_grad_norm[2]
        self.last_grad = []
        self.last_grad_noisy = []
        self.global_last_grad = []
        # self.kfilter = KalmanFilter(0, self.max_grad_norm**2, self.max_grad_norm**2)
        self.kfilter = None
        self.log = []

        for p in self.params:
            p.summed_grad = None


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
        # init
        if self.last_grad == []:
            self.add_noise()
            self.kfilter.x = [p.grad/len(self.grad_samples[0]) for p in self.params]
            self.last_grad = self.kfilter.x
            self.log = [[torch.mean(g, dim=0) for g in self.grad_samples], self.last_grad, self.last_grad] # clean, noisy, estimate

        # filter
        else:
            if self.log == []:
                self.log = [[torch.mean(g, dim=0) for g in self.grad_samples], self.last_grad, self.last_grad] # clean, noisy, estimate
            ns = len(self.grad_samples[0])
            self.KFpredict()
            self.log[0] = [p.summed_grad/ns for p in self.params]
            self.add_noise()
            self.log[1] = [p.grad/ns for p in self.params]
            self.KFcorrect()
            self.log[2] = self.last_grad

        self.scale_grad()

        if self.step_hook:
            self.step_hook(self)

        self._is_last_step_skipped = False
        return True  
    
    def KFpredict(self):
        self.kfilter.predict()

    def KFcorrect(self):
        num_samples = len(self.grad_samples[0])
        z = [p.grad/num_samples for p in self.params]
        x_hat = self.kfilter.correct(z)
        for i in range(len(self.params)):
            self.params[i].grad = x_hat[i] * num_samples
        self.last_grad = x_hat

    def add_noise(self):
        """
        Adds noise to clipped gradients. Stores clipped and noised result in ``p.grad``
        """
        for p in self.params:
            _check_processed_flag(p.summed_grad)

            p.grad = (p.summed_grad).view_as(p)

            _mark_as_processed(p.summed_grad)


class KFilterDPOptimizer(KFilterOptimizer):
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

        self.max_grad_norm = max_grad_norm[0]
        self.perp_grad_norm = max_grad_norm[1]
        self.rate_dr = max_grad_norm[2]
        self.last_grad = []
        self.last_grad_noisy = []
        self.global_last_grad = []
        self.log = []

        for p in self.params:
            p.summed_grad = None

    def add_noise(self):
        """
        Adds noise to clipped gradients. Stores clipped and noised result in ``p.grad``
        """
        for p in self.params:
            _check_processed_flag(p.summed_grad)

            noise = _generate_noise(
                std=self.noise_multiplier * self.max_grad_norm,
                reference=p.summed_grad,
                generator=self.generator,
                secure_mode=self.secure_mode,
            )
            p.grad = (p.summed_grad + noise).view_as(p)

            _mark_as_processed(p.summed_grad)

