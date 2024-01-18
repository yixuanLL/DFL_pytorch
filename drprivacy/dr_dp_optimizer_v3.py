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
class DrDPOptimizerV3(DPOptimizer):
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
        self.clip_paral =  max_grad_norm[3]
        self.last_grad = []
        self.last_normratio = []
        self.norm = 1
        self.steps = 1
        self.global_last_grad = []
        self.g_perp_sum = []
        self.cos_sum = []
        self.log = []
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
        gi_perp, alpha_i = self.decompose_grad()   
        g_perp = self.clip_g_perp(gi_perp) 
        g_perp_clean = copy.deepcopy(g_perp) 
        self.add_noise_sum(g_perp, self.noise_multiplier, self.perp_grad_norm)

        # preserve paral factor
        if self.last_grad != []:
            clip_p = self.clip_paral
            alpha = self.clip(alpha_i, clip_p)
            alpha_clean = copy.deepcopy(alpha)
            self.add_noise_sum(alpha, self.noise_multiplier_2, clip_p) 
        else:
            alpha = 0
            alpha_clean = 0
        g_perp = self.recover_grad(g_perp, alpha) 


    def decompose_grad(self):
        if self.last_grad == []:      
            print('DPDR V3')
            return self.grad_samples, [torch.tensor(1.).to(self.device)]*8
        last_grad_norms = [g.reshape(-1).norm(2, dim=-1) for g in self.last_grad] # norm of per laryer of last gradient
        paral_alpha = [torch.sum(g.reshape(len(g), -1)*(lg.reshape(-1)), dim=1)/(lg_norm*lg_norm) for (g, lg, lg_norm) in zip(self.grad_samples, self.last_grad, last_grad_norms)]
        gi_paral = [paral.reshape([len(self.grad_samples[0])]+[1]*len(lg.shape)) * torch.tile(lg.unsqueeze(0),[len(self.grad_samples[0])]+[1]*len(lg.shape)) for paral, lg in zip(paral_alpha, self.last_grad)]
        gi_perp = [(g-gl) for g, gl in zip(self.grad_samples, gi_paral)] 
        return gi_perp, paral_alpha

    # def recover_grad(self, g_perp, g_perp_noisy, costheta, costheta_noisy):
    def recover_grad(self, g_perp_noisy, alpha_noisy):
        if self.last_grad == []:
            g_noisy = g_perp_noisy
            g = g_perp_noisy
        else:
            g_noisy = [gp + a * lg for gp, a, lg in zip(g_perp_noisy, alpha_noisy, self.last_grad)]
        for p,gi in zip(self.params, g_noisy):
            if p.summed_grad is not None:
                p.summed_grad += gi
            else:
                p.summed_grad = gi
        self.last_grad = g_noisy

        # for historical grad
        # noisy_mean_g = [p.summed_grad/len(self.grad_samples[0]) for p in self.params]
        # if self.steps == 1: # accumulation
        #     self.last_grad_noisy = noisy_mean_g
        # else:
        #     self.last_grad_noisy = [(g+lg*self.steps)/(self.steps+1) for g, lg in zip(noisy_mean_g, self.last_grad_noisy)]
        
        return g_perp_noisy


    def clip_g_perp(self, g_perp):
        per_param_norms = [g.reshape(len(g), -1).norm(2, dim=-1) for g in g_perp] # norm of per laryer of per sample gradient
        per_sample_norms = torch.stack(per_param_norms, dim=1).norm(2, dim=1) # norm of per sample gradient
        per_sample_clip_factor = (self.perp_grad_norm / (per_sample_norms + 1e-6)).clamp(max=1.0) # clip [ max min ]

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
            # print('pre clip grad norm', per_param_norms[0][0])
        for p in self.params:
            _check_processed_flag(p.grad_sample)
            grad_sample = self._get_flat_grad_sample(p)
            # grad = contract("i,i...", per_sample_clip_factor, grad_sample)
            p.grad_sample = torch.reshape(per_sample_clip_factor, [len(grad_sample)]+[1]*(len(grad_sample.shape)-1)) * grad_sample
            _mark_as_processed(p.grad_sample)

        # t = [g.reshape(len(g), -1).norm(2, dim=-1) for g in self.grad_samples]
        # tt = torch.stack(t, dim=1).norm(2, dim=1)
        # print('after clip grad norm', per_param_norms[0][0])
        # print(1)


    def clip(self, vec, clip_bound):
        # print(vec)
        # norm = vec
        norm = torch.stack(vec, dim=1).norm(2, dim=1)
        clip_factor = (
            clip_bound / (norm + 1e-6)
        ).clamp(max=1.0)
        vec = [torch.sum(clip_factor * v) for v in vec]
        return vec

    def add_noise(self):
        """
        Adds noise to clipped gradients. Stores clipped and noised result in ``p.grad``
        """
        for p in self.params:
            _check_processed_flag(p.summed_grad)
            p.grad = (p.summed_grad).view_as(p)
            _mark_as_processed(p.summed_grad)

    # def add_noise_mean(self, cos, noise_multiplier, sensitivity): 
    #     """
    #     Adds noise to clipped gradients. Stores clipped and noised result in ``p.grad``
    #     """
    #     std = noise_multiplier * sensitivity
    #     std /= (len(self.grad_samples[0]))
    #     for c in cos:
    #         noise = torch.normal(
    #         mean=0,
    #         std=std,
    #         size=c.shape,
    #         device=self.device,
    #         generator=None,
    #     )
    #         c += noise
    #     return cos

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
      
class DrKFDPOptimizertest2(DrDPOptimizerV3): # add KF filter
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
        self.clip_paral =  max_grad_norm[3]
        self.last_grad = []
        self.last_grad_noisy = []
        self.last_normratio = []
        self.norm = 1
        self.global_last_grad = []
        self.g_perp_sum = []
        self.cos_sum = []
        self.log = []
        self.steps = 1
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
        
        
        if self.last_grad == []:
            self.dr_process()
            self.add_noise()
            self.kfilter.x = self.last_grad_noisy 
            self.log = [[torch.mean(g, dim=0) for g in self.grad_samples], self.last_grad_noisy, self.last_grad_noisy] # clean, noisy, estimate
 
        else:        
            if self.log == []:
                self.log = [[torch.mean(g, dim=0) for g in self.grad_samples], self.last_grad_noisy, self.last_grad_noisy] # clean, noisy, estimate
            # self.KFpredict()
            self.dr_process()
            self.add_noise()
            # self.log[0:2] = [self.last_grad, self.last_grad_noisy]
            # self.KFcorrect()

        self.scale_grad()


        if self.step_hook:
            self.step_hook(self)

        self._is_last_step_skipped = False
        return True   
      
    def KFpredict(self):
        self.kfilter.predict()

    def KFcorrect(self):
        num_samples = len(self.grad_samples[0])
        z = self.last_grad_noisy
        g_estimate = self.kfilter.correct(z) # g_estimate
        for i in range(len(self.params)):
            self.params[i].grad = g_estimate[i] * num_samples
        self.log[2] = g_estimate
        self.last_grad = [g/torch.norm(g, keepdim=False) for g in self.last_grad]
        self.last_grad_noisy = [gi/torch.norm(gi, keepdim=False) for gi in g_estimate]

    # use this function: without diff2; no this function: use diff2
    def decompose_grad(self):
        if self.last_grad == []:      
            return self.grad_samples, [torch.tensor(1.).to(self.device)]*8
        last_grad_noisy = [gi/torch.norm(gi, keepdim=False) for gi in self.last_grad_noisy]
        # to test my proof by using clean last_grad
        # last_grad_noisy = [gi/torch.norm(gi, keepdim=False) for gi in self.last_grad]
        ## end test
        per_param_norms = [g.reshape(len(g), -1).norm(2, dim=-1) for g in self.grad_samples] # norm of per laryer of per sample gradient
        last_grad_norms = [g.reshape(-1).norm(2, dim=-1) for g in last_grad_noisy] # norm of per laryer of last gradient
        costheta = [torch.mean(torch.sum(g.reshape(len(g), -1)*(lg.reshape(-1)), dim=1)/(g_norm*lg_norm)) for (g, lg, g_norm, lg_norm) in zip(self.grad_samples, last_grad_noisy, per_param_norms, last_grad_norms)]
        gi_paral = [torch.reshape(gn*cos, [len(gn)]+[1]*len(lg.shape)) * torch.tile(lg.unsqueeze(0),[len(gn)]+[1]*len(lg.shape)) for gn, cos, lg in zip(per_param_norms, costheta, last_grad_noisy)]
        gi_perp = [(g-gl) for g, gl in zip(self.grad_samples, gi_paral)] 
        paral_alpha =  [torch.mean(gn*cos, dim=0) for cos, gn in zip(per_param_norms, costheta)]
        return gi_perp, paral_alpha 

    def recover_grad(self, g_perp, g_perp_noisy, g_norm, g_norm_noisy):
        if self.last_grad == []:
            g_noisy = g_perp_noisy
            g = g_perp_noisy
            self.last_grad_noisy = g_noisy
        else:
            g_noisy = [ gp + gn * lg * len(self.grad_samples[0]) for gp, gn, lg in zip(g_perp_noisy, g_norm_noisy, self.last_grad_noisy)]
        for p,gi in zip(self.params, g_noisy):
            if p.summed_grad is not None:
                p.summed_grad += gi
            else:
                p.summed_grad = gi
        # gt = [torch.sum(p.grad_sample, dim=0) for p in self.params]
        self.last_grad = [torch.mean(g, dim=0) for g in self.grad_samples]
        self.last_grad_noisy = [p.summed_grad/len(self.grad_samples[0]) for p in self.params]
        return g_perp             
