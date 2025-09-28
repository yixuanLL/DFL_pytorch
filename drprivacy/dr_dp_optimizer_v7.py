from opacus.optimizers import DPOptimizer
from opacus.optimizers.optimizer import _check_processed_flag, _generate_noise, _mark_as_processed
from torch.optim import Optimizer
from typing import Callable, List, Optional, Union
import torch
from opt_einsum.contract import contract
import copy
# from utils.dpsgd_utils import exp_topk
import math

# append alpha with g_perp as the d+m dimension vector, clip together with C, decompose and add weight separately, perturb with the same sigma together
# no early stop
class DrDPOptimizerV7(DPOptimizer):
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
        print('===Dr DP v7===')
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
        self.noise_multiplier_a = max_grad_norm[2]
        self.alpha_norm =  max_grad_norm[3]
        self.noise_multiplier_g = max_grad_norm[4]
        self.steps_interval = max_grad_norm[5]
        self.steps_dr = max_grad_norm[6]
        self.weight_alpha = max_grad_norm[7]
        self.last_grad = []
        self.last_normratio = []
        self.norm = 1
        self.steps = 1
        self.global_last_grad = []
        self.g_perp_sum = None
        self.alpha_sum = None
        self.log = []
        self.s = 196
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

        # self.clip_and_accumulate()
        # if self.steps % self.steps_interval < self.steps_dr:
        if 1==1: # always dpdr, no early stop
            gi_perp, alpha_i = self.dr_process() 
            self.clip_combined_vec(gi_perp, alpha_i, self.max_grad_norm)
            if self._check_skip_next_step():
                self._is_last_step_skipped = True
                return False

            self.add_noise_sum(self.g_perp_sum, self.noise_multiplier_g, self.max_grad_norm)
            self.add_noise_sum(self.alpha_sum, self.noise_multiplier_g, self.max_grad_norm) 
            self.recover_grad(self.g_perp_sum, self.alpha_sum) 
        else: 
            norm_dpsgd = 0.2 #=cifar10 dpsgd clipping bound C_g
            self.dpsgd(norm_dpsgd) 
            if self._check_skip_next_step():
                self._is_last_step_skipped = True
                return False
            self.add_noise_dpsgd(norm_dpsgd)

        self.scale_grad()
        # self.log = [[torch.mean(g, dim=0) for g in self.grad_samples], self.last_grad, []]

        if self.step_hook:
            self.step_hook(self)

        self._is_last_step_skipped = False
        return True  
    
    def dpsgd(self, norm):
        g = self.clip_g_perp(self.grad_samples, norm)
        noisy_mean_g = copy.deepcopy([gg/len(self.grad_samples[0]) for gg in g]) 
        # self.last_grad = noisy_mean_g

        for r,p in zip(g,self.params):
            if p.summed_grad is not None:
                p.summed_grad += r
            else:
                p.summed_grad = r

        if self.steps % self.s <= 150:
        # if self.steps == 1: # accumulation
            self.last_grad = noisy_mean_g
        else:
            self.last_grad = [(g+lg*(self.steps%self.s))/(self.steps%self.s+1) for g, lg in zip(noisy_mean_g, self.last_grad)]
            # normalize
            last_norm = [p.reshape(-1).norm(2, dim=-1) for p in self.last_grad]
            norm = torch.stack(last_norm).norm(2)
            self.last_grad = [p/norm for p in self.last_grad]
        return g

    def dr_process(self):
        gi_perp, alpha_i = self.decompose_grad()   
        gi_perp, alpha_i = self.add_weight(gi_perp, alpha_i)
        return gi_perp, alpha_i

        # sum up
        # g_perp_summed = []
        # for p in gi_perp:
        #     grad = contract("i,i...", 1, p) # mutiply [128] * [128, 16, 1, 8, 8] -> [16, 1, 8, 8] sum
        #     g_perp_summed.append(grad)
        # if self.g_perp_sum is not None:
        #     self.g_perp_sum = copy.deepcopy([sum+gp for sum, gp in zip(self.g_perp_sum, g_perp_summed)])
        # else:
        #     self.g_perp_sum = copy.deepcopy(g_perp_summed)

        # vec = [torch.sum(v) for v in alpha_i]
        # if self.alpha_sum is not None:
        #     self.alpha_sum = copy.deepcopy([sum+gp for sum, gp in zip(self.alpha_sum, vec)])
        # else:
        #     self.alpha_sum = copy.deepcopy(vec)


    def decompose_grad(self):
        paral_alpha = [torch.sum(g.reshape(len(g), -1)*(lg.reshape(-1)), dim=1) for (g, lg) in zip(self.grad_samples, self.last_grad)] # norm of last grad is 1; shape=[8, batch_size]
        gi_paral = [paral.reshape([len(self.grad_samples[0])]+[1]*len(lg.shape)) * torch.tile(lg.unsqueeze(0),[len(self.grad_samples[0])]+[1]*len(lg.shape)) for paral, lg in zip(paral_alpha, self.last_grad)]
        gi_perp = [(g-gl) for g, gl in zip(self.grad_samples, gi_paral)] 

        # per_param_norms = [g.reshape(len(g), -1).norm(2, dim=-1) for g in gi_perp] # norm of per laryer of per sample gradient
        # gi_perp_norm = torch.stack(per_param_norms, dim=1).norm(2, dim=1)

        # per_param_norms = [g.reshape(len(g), -1).norm(2, dim=-1) for g in gi_paral] # norm of per laryer of per sample gradient
        # gi_para_norm = torch.stack(per_param_norms, dim=1).norm(2, dim=1)
        return gi_perp, paral_alpha


    def add_weight(self, gi_perp, paral_alpha):
        alphai_norm = torch.stack(paral_alpha, dim=1).norm(2, dim=1)
        # weight_perp_i = (self.max_grad_norm **2 - self.weight_alpha * alpha_norm**2) / (self.max_grad_norm **2 - alpha_norm**2) # a tensor not a scalar
        per_param_norms = [g.reshape(len(g), -1).norm(2, dim=-1) for g in gi_perp] # norm of per laryer of per sample gradient
        gi_perp_norm = torch.stack(per_param_norms, dim=1).norm(2, dim=1) # norm of per sample gradient
        # need check weight_perp_i, expected shape [batchsize,]
        weight_perp_i = [torch.sqrt(1 + (1-self.weight_alpha**2) * an/gpn)for an,gpn in zip(alphai_norm, gi_perp_norm)] # each user has one weight
        # print(weight_perp_i)
        gi_perp = [g*w for g,w in zip(gi_perp, weight_perp_i)]
        paral_alpha = [g*self.weight_alpha for g in paral_alpha]
        return gi_perp, paral_alpha

    def recover_grad(self, g_perp_noisy, alpha_noisy):
        g_noisy = [gp + a * lg for gp, a, lg in zip(g_perp_noisy, alpha_noisy, self.last_grad)]
        for g,p in zip(g_noisy, self.params):
            _check_processed_flag(p.summed_grad)
            p.grad = (g).view_as(p)
            _mark_as_processed(p.summed_grad)

        # self.last_grad = copy.deepcopy([g/len(self.grad_samples[0]) for g in g_noisy]) 
        # for historical grad
        noisy_mean_g = copy.deepcopy([g/len(self.grad_samples[0]) for g in g_noisy])

        # if self.steps % self.s == 0: # accumulation
        if False:
            self.last_grad = noisy_mean_g
        else:
            self.last_grad = [(g+lg*(self.steps%self.s))/(self.steps%self.s+1) for g, lg in zip(noisy_mean_g, self.last_grad)]
            # normalize
            last_norm = [p.reshape(-1).norm(2, dim=-1) for p in self.last_grad]
            norm = torch.stack(last_norm).norm(2)
        self.last_grad = [p/norm for p in self.last_grad]

        self.g_perp_sum = None
        self.alpha_sum = None



    # def clip_g_perp(self, g_perp, clip_norm):
    #     per_param_norms = [g.reshape(len(g), -1).norm(2, dim=-1) for g in g_perp] # norm of per laryer of per sample gradient
    #     per_sample_norms = torch.stack(per_param_norms, dim=1).norm(2, dim=1) # norm of per sample gradient
    #     per_sample_clip_factor = (clip_norm / (per_sample_norms + 1e-6)).clamp(max=1.0) # clip [ max min ]
    #     g_perp_clipped = []
    #     for p in g_perp:
    #         grad = contract("i,i...", per_sample_clip_factor, p) # mutiply [128] * [128, 16, 1, 8, 8] -> [16, 1, 8, 8] clip & sum
    #         g_perp_clipped.append(grad)
    #     if self.g_perp_sum is not None:
    #         self.g_perp_sum = copy.deepcopy([sum+gp for sum, gp in zip(self.g_perp_sum, g_perp_clipped)])
    #     else:
    #         self.g_perp_sum = copy.deepcopy(g_perp_clipped)
    #     return g_perp_clipped

    def clip_combined_vec(self, g_perp, alpha, clip_norm): #[g_perp_1, ..., g_perp_n, alpha_1, ..., alpha_m]
        per_param_norms = [g.reshape(len(g), -1).norm(2, dim=-1) for g in g_perp] # norm of per laryer of per sample gradient
        g_perp_i_norms = torch.stack(per_param_norms, dim=1).norm(2, dim=1) # norm of per sample gradient
        alpha_i_norms = torch.stack(alpha, dim=1).norm(2, dim=1)
        
        per_vec_clip_factor = (clip_norm / (alpha_i_norms + g_perp_i_norms + 1e-6)).clamp(max=1.0) # clip [ max min ]
        g_perp_clipped = []
        for p in g_perp:
            grad = contract("i,i...", per_vec_clip_factor, p) # mutiply [128] * [128, 16, 1, 8, 8] -> [16, 1, 8, 8] clip & sum
            g_perp_clipped.append(grad)
        if self.g_perp_sum is not None:
            self.g_perp_sum = copy.deepcopy([sum+gp for sum, gp in zip(self.g_perp_sum, g_perp_clipped)])
        else:
            self.g_perp_sum = copy.deepcopy(g_perp_clipped)

        alpha_clipped = []
        for a in alpha:
            tmp = contract("i,i...", per_vec_clip_factor, a) # mutiply [128] * [128, 16, 1, 8, 8] -> [16, 1, 8, 8] clip & sum
            alpha_clipped.append(tmp)
        if self.alpha_sum is not None:
            self.alpha_sum = copy.deepcopy([sum+gp for sum, gp in zip(self.g_perp_sum, g_perp_clipped)])
        else:
            self.alpha_sum = copy.deepcopy(alpha_clipped)
        return g_perp_clipped, alpha_clipped

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
            p.grad_sample = torch.reshape(per_sample_clip_factor, [len(grad_sample)]+[1]*(len(grad_sample.shape)-1)) * grad_sample # grad of each user

    # def clip_alpha(self, vec, clip_bound):
    #     norm = torch.stack(vec, dim=1).norm(2, dim=1)
    #     clip_factor = (
    #         clip_bound / (norm + 1e-6)
    #     ).clamp(max=1.0)
    #     vec = [torch.sum(clip_factor * v) for v in vec]
    #     if self.alpha_sum is not None:
    #         self.alpha_sum = copy.deepcopy([sum+gp for sum, gp in zip(self.alpha_sum, vec)])
    #     else:
    #         self.alpha_sum = copy.deepcopy(vec)
    #     # return vec

    def add_noise_dpsgd(self,norm_dpsgd):
        """
        Adds noise to clipped gradients. Stores clipped and noised result in ``p.grad``
        """
        std = self.noise_multiplier_g * norm_dpsgd
        for p in self.params:
            _check_processed_flag(p.summed_grad)

            noise = torch.normal(
            mean=0,
            std=std,
            size=p.summed_grad.shape,
            device=self.device,
            generator=None,
            )
            p.grad = (p.summed_grad+noise).view_as(p)
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
