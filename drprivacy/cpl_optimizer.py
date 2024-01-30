from opacus.optimizers import DPOptimizer
from opacus.optimizers.optimizer import _check_processed_flag, _generate_noise, _mark_as_processed
from torch.optim import Optimizer
from typing import Callable, List, Optional, Union
import torch
from opt_einsum.contract import contract
import copy
from utils.dpsgd_utils import exp_topk
from utils.kalman_filter import KalmanFilter

class CplOptimizer(DPOptimizer):
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
        self.last_grad_noisy = []
        self.global_last_grad = []
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
        a = copy.deepcopy(self.last_grad[0]*127)        
        self.complement_process()
        
        self.add_noise()

        b = copy.deepcopy(self.last_grad[0])
        # print('last grad:', torch.sum(a))
        # print('delta last grad:', torch.sum(b-a)/128.0)

        self.scale_grad()

        if self.step_hook:
            self.step_hook(self)

        self._is_last_step_skipped = False
        return True  

    def complement_process(self): # baseline V.S. dr_process
        if self.last_grad == []:
            delta_g = [p.grad_sample for p in self.params]
        else:
            delta_g = [p.grad_sample - torch.tile(lg.unsqueeze(0),[len(p.grad_sample)]+[1]*len(lg.shape)) for lg, p  in zip(self.last_grad, self.params)]    
        gi_cpl_clipped = self.clip_g_perp(delta_g)
        # print([torch.norm(g, keepdim=False) for g in gi_cpl_clipped])

        g_reverse = self.reverse_process(gi_cpl_clipped)
        # g_reverse = [torch.sum(g, dim=0) for g in gi_reverse]

        for p,gi in zip(self.params, g_reverse):
            if p.summed_grad is not None:
                p.summed_grad += gi
            else:
                p.summed_grad = gi
        return
    
    def reverse_process(self, gi_cpl):
        if self.last_grad == []:
            return gi_cpl
        g_reverse = [gc + lg*len(self.grad_samples[0]) for gc, lg in zip(gi_cpl, self.last_grad)]
        # g_reverse = [lg*len(self.grad_samples[0]) for gc, lg in zip(gi_cpl, self.last_grad)]
        # print('delta/last_grad', torch.sum(gi_cpl[0])/torch.sum(self.last_grad[0]*128))
        return g_reverse


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
 
            _mark_as_processed(p.grad_sample)


    def add_noise(self):
        """
        Adds noise to clipped gradients. Stores clipped and noised result in ``p.grad``
        """

        for p in self.params:
            _check_processed_flag(p.summed_grad)

            p.grad = (p.summed_grad).view_as(p)

            _mark_as_processed(p.summed_grad)
        # self.last_grad = [p.grad for p in self.params] 
        # self.last_grad = g_noisy # wrong
        self.last_grad = copy.deepcopy([p.grad/len(self.grad_samples[0]) for p in self.params]) 
        # accumulative gradients
        # mean_g = [p.grad/len(p.grad_sample) for p in self.params] 
        # if self.steps == 1:
        #     self.last_grad = mean_g
        # else:
        #     self.last_grad = [(g+lg*self.steps)/(self.steps+1) for g, lg in zip(mean_g, self.last_grad)]
 


    def clip_g_perp(self, g_perp):
        per_param_norms = [
            g.reshape(len(g), -1).norm(2, dim=-1) for g in g_perp
        ] # norm of per laryer of per sample gradient
        per_sample_norms = torch.stack(per_param_norms, dim=1).norm(2, dim=1) # norm of per sample gradient
        # print(per_sample_norms[0:5])
        per_sample_clip_factor = (
            self.perp_grad_norm / (per_sample_norms + 1e-6)
        ).clamp(max=1.0) # clip [ max min ]

        g_perp_clipped = []
        for p in g_perp:
            # grad_sample = self._get_flat_grad_sample(p) # change in to one tensor
            grad = contract("i,i...", per_sample_clip_factor, p) # mutiply [128] * [128, 16, 1, 8, 8] -> [16, 1, 8, 8] clip & sum
            # grad = torch.reshape(per_sample_clip_factor, [len(p)]+[1]*(len(p.shape)-1)) * p
            g_perp_clipped.append(grad)
        return g_perp_clipped

class CplDPOptimizer(CplOptimizer):
# class CplDPOptimizer_bak(CplOptimizer):
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
        # super(CplOptimizer, self).__init__(optimizer, noise_multiplier, max_grad_norm, expected_batch_size, loss_reduction, generator, secure_mode)
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
        self.global_last_grad = []
        self.log = []
        self.steps = 1
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def complement_process(self): # baseline V.S. dr_process
        if self.last_grad == []:
            delta_g = [p.grad_sample for p in self.params]
        else:
            delta_g = [p.grad_sample - torch.tile(lg.unsqueeze(0),[len(p.grad_sample)]+[1]*len(lg.shape)) for lg, p  in zip(self.last_grad, self.params)]    
        g_cpl_clipped = self.clip_g_perp(delta_g)
        self.add_noise_sum(g_cpl_clipped, self.noise_multiplier, self.perp_grad_norm)
        # print([torch.norm(g, keepdim=False) for g in g_cpl_clipped])


        g_reverse = self.reverse_process(g_cpl_clipped)
        # g_reverse = [torch.sum(g, dim=0) for g in gi_reverse]

        for p,gi in zip(self.params, g_reverse):
            if p.summed_grad is not None:
                p.summed_grad += gi
            else:
                p.summed_grad = gi
        return
            
    def add_noise_sum(self, vec, noise_multiplier, sensitivity):
        """
        Adds noise to clipped gradients. Stores clipped and noised result in ``p.grad``
        """
        std = noise_multiplier * sensitivity
        for i, v in enumerate(vec):
            noise = torch.normal(
            mean=0,
            std=std,
            size=v.shape,
            device=self.device,
            generator=None,
        )
            v += noise
            # if i==0:
            #     print('noise:',torch.sum(noise))
        return vec
    # def add_noise(self):
    #     """
    #     Adds noise to clipped gradients. Stores clipped and noised result in ``p.grad``
    #     """
    #     # self.last_grad = [p.summed_grad/len(p.grad_sample) for p in self.params] # self.last_grad采用clean gradients
    #     # self.last_grad = [for p in self.grad_samples]# self.last_grad采用robust gradients
    #     # if self.last_grad == []:
    #     #     print('DPcpl')
    #     for p in self.params:
    #         _check_processed_flag(p.summed_grad)
    #         # print(torch.norm(p.summed_grad))
    #         noise = _generate_noise(
    #             std=self.noise_multiplier * self.perp_grad_norm,
    #             reference=p.summed_grad,
    #             generator=self.generator,
    #             secure_mode=self.secure_mode,
    #         )
    #         p.grad = (p.summed_grad + noise).view_as(p)
    #         # print(torch.norm(p.grad))
    #         _mark_as_processed(p.summed_grad)
    #     self.last_grad = [p.grad for p in self.params] 
    #     # accumulative gradients
    #     # mean_g = [p.grad for p in self.params] 
    #     # if self.steps == 1:
    #     #     self.last_grad = mean_g
    #     # else:
    #     #     self.last_grad = [(g+lg*self.steps)/(self.steps+1) for g, lg in zip(mean_g, self.last_grad)]


class CplKFDPOptimizer(CplDPOptimizer):
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
        # super(CplOptimizer, self).__init__(optimizer, noise_multiplier, max_grad_norm, expected_batch_size, loss_reduction, generator, secure_mode)
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
        self.global_last_grad = []
        self.log = []
        self.steps = 1

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
            self.complement_process()
            self.add_noise()
            self.kfilter.x = self.last_grad  
        else:
            self.KFpredict()
            self.complement_process()
            self.add_noise()
            self.KFcorrect()

        self.scale_grad()

        if self.step_hook:
            self.step_hook(self)

        self._is_last_step_skipped = False
        return True 

    def KFpredict(self):
        self.kfilter.predict()

    def KFcorrect(self):
        num_samples = len(self.grad_samples[0])
        z = self.last_grad
        g_estimate = self.kfilter.correct(z) # g_estimate
        # for i in range(len(self.params)):
        #     self.params[i].grad = g_estimate[i] * num_samples
        self.last_grad = g_estimate

# class CplDPOptimizer(CplOptimizer): #similar with DIFF2; use clean gradient for complementary calculation
class diff2(CplOptimizer):
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
        # super(CplOptimizer, self).__init__(optimizer, noise_multiplier, max_grad_norm, expected_batch_size, loss_reduction, generator, secure_mode)
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
        self.last_grad_noisy = []
        self.global_last_grad = []
        self.log = []
        # self.kfilter = KalmanFilter(0, self.max_grad_norm**2, (self.perp_grad_norm* self.noise_multiplier/self.expected_batch_size)**2)
        self.kfilter = KalmanFilter(0, (self.max_grad_norm*0.1)**2, (self.perp_grad_norm*self.noise_multiplier/self.expected_batch_size)**2)

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
            self.complement_process()
            self.add_noise()
            self.kfilter.x = self.last_grad_noisy  
        else:
            self.KFpredict()
            self.complement_process()
            self.add_noise()
            self.KFcorrect()

        self.scale_grad()

        if self.step_hook:
            self.step_hook(self)

        self._is_last_step_skipped = False
        return True 


    def reverse_process(self, gi_cpl):
        if self.last_grad == []:
            return gi_cpl
        g_reverse = [gc + lg*len(self.grad_samples[0]) for gc, lg in zip(gi_cpl, self.last_grad_noisy)]
        # g_reverse = [gc + lg*len(self.grad_samples[0]) for gc, lg in zip(gi_cpl, self.last_grad)] #看denoise的空间还有多大
        return g_reverse 

    def add_noise(self):
        """
        Adds noise to clipped gradients. Stores clipped and noised result in ``p.grad``
        """
        # self.last_grad = [p.summed_grad/len(p.grad_sample) for p in self.params] # clean gradients based on reversed gradient
        self.last_grad = [torch.mean(g, dim=0) for g in self.grad_samples] # clean gradients based on current gradient
        for p in self.params:
            _check_processed_flag(p.summed_grad)
            # print(torch.norm(p.summed_grad))
            noise = _generate_noise(
                std=self.noise_multiplier * self.perp_grad_norm,
                reference=p.summed_grad,
                generator=self.generator,
                secure_mode=self.secure_mode,
            )
            p.grad = (p.summed_grad + noise).view_as(p)

            _mark_as_processed(p.summed_grad)

        self.last_grad_noisy = [p.grad for p in self.params] # noisy gradients

    def KFpredict(self):
        self.kfilter.predict()

    def KFcorrect(self):
        num_samples = len(self.grad_samples[0])
        z = self.last_grad_noisy
        g_estimate = self.kfilter.correct(z) # g_estimate
        for i in range(len(self.params)):
            self.params[i].grad = g_estimate[i] * num_samples
        self.last_grad_noisy = g_estimate


        
            
