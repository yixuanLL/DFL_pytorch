# -*- coding: utf-8 -*-
# @Author : Zhang
# @Email : zl16035056@163.com
# @File : dpsgd_utils.py

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import re
import numpy as np
try:
    import torch
except ImportError:
    torch = None
try:
    from scipy import special
except ImportError:
    class _SpecialFallback:
        @staticmethod
        def binom(n, k):
            if k < 0 or k > n:
                return 0.0
            return math.exp(
                math.lgamma(n + 1.0)
                - math.lgamma(k + 1.0)
                - math.lgamma(n - k + 1.0)
            )

        @staticmethod
        def ndtr(x):
            return 0.5 * math.erfc(-x / math.sqrt(2.0))

        @staticmethod
        def log_ndtr(x):
            if x < -10.0:
                return -0.5 * x * x - math.log(-x) - 0.5 * math.log(2.0 * math.pi)
            p = _SpecialFallback.ndtr(x)
            return -np.inf if p == 0.0 else math.log(p)

    special = _SpecialFallback()
# from opacus.accountants.utils import get_noise_multiplier
# from opacus import PrivacyEngine
np.random.seed(10)

def _count_dpdr_steps(total_steps, steps_dr=None, steps_interval=None):
    """Count GDR steps using DrDPOptimizerV5's actual step schedule."""
    if total_steps <= 0:
        return 0
    if steps_dr is None:
        return total_steps
    if steps_dr <= 0:
        return 0
    if steps_interval is None or steps_interval <= 0:
        return min(steps_dr, total_steps)
    if steps_dr >= steps_interval:
        return total_steps

    full_cycles, remainder = divmod(total_steps, steps_interval)
    # Optimizer steps are counted from 1, and GDR is used when
    # step % steps_interval < steps_dr. This includes the cycle boundary
    # step where modulo is 0.
    gdr_per_full_cycle = steps_dr
    gdr_steps = full_cycles * gdr_per_full_cycle
    for step_mod in range(1, remainder + 1):
        if step_mod % steps_interval < steps_dr:
            gdr_steps += 1
    return min(gdr_steps, total_steps)


def _gdp_delta_at_zero(mu):
    if mu <= 0:
        return 0.0
    return 2.0 * special.ndtr(-mu / 2.0)


def gdp_delta(epsilon, mu):
    """Convert Gaussian DP mu to delta at a given epsilon.

    delta(eps; mu) = Phi(-eps/mu + mu/2)
                     - exp(eps) Phi(-eps/mu - mu/2)
    """
    if epsilon < 0:
        raise ValueError("epsilon must be non-negative.")
    if mu < 0:
        raise ValueError("mu must be non-negative.")
    if mu == 0:
        return 0.0
    if math.isinf(mu):
        return 1.0

    log_term1 = special.log_ndtr(-epsilon / mu + mu / 2.0)
    log_term2 = epsilon + special.log_ndtr(-epsilon / mu - mu / 2.0)
    if log_term2 >= log_term1:
        return 0.0
    return float(math.exp(_log_sub(log_term1, log_term2)))


def gdp_epsilon_from_mu(mu, delta):
    """Return the smallest epsilon whose GDP delta is at most delta."""
    if delta <= 0:
        raise ValueError("delta must be positive.")
    if mu < 0:
        raise ValueError("mu must be non-negative.")
    if mu == 0:
        return 0.0
    if math.isinf(mu):
        return np.inf
    if delta >= _gdp_delta_at_zero(mu):
        return 0.0

    low, high = 0.0, 1.0
    while gdp_delta(high, mu) > delta:
        high *= 2.0
        if high > 1e12:
            return np.inf

    for _ in range(100):
        mid = (low + high) / 2.0
        if gdp_delta(mid, mu) > delta:
            low = mid
        else:
            high = mid
    return high


def gdp_mu_from_epsilon(epsilon, delta):
    """Invert GDP accounting: find the largest mu satisfying (epsilon, delta)."""
    if epsilon < 0:
        raise ValueError("epsilon must be non-negative.")
    if delta <= 0:
        raise ValueError("delta must be positive.")
    if epsilon == 0:
        return 0.0

    low, high = 0.0, 1.0
    while gdp_delta(epsilon, high) <= delta:
        low = high
        high *= 2.0
        if high > 1e6:
            return np.inf

    for _ in range(100):
        mid = (low + high) / 2.0
        if gdp_delta(epsilon, mid) <= delta:
            low = mid
        else:
            high = mid
    return low


def _poisson_gdp_term(mu0):
    if mu0 == 0:
        return 0.0
    if math.isinf(mu0):
        return np.inf
    try:
        return math.expm1(mu0 ** 2)
    except OverflowError:
        return np.inf


def compute_dpdr_gdp_mu(local_dataset_size, local_batch_size, epochs,
                        sigma_perp, sigma_alpha, sigma_g,
                        steps_dr=None, steps_interval=None,
                        return_details=False):
    """Compute Poisson-GDP mu for DPDR with joint Gaussian GDR accounting.

    DPDR's GDR stage releases the perpendicular component and alpha jointly,
    so the per-step Gaussian-DP parameter is
    sqrt(sigma_perp^-2 + sigma_alpha^-2). The gradient dimension d and
    projection dimension m do not enter this privacy mu; dimensions only affect
    utility/convergence through variance.
    """
    if local_dataset_size <= 0 or local_batch_size <= 0:
        raise ValueError("dataset size and batch size must be positive.")

    total_steps = int(local_dataset_size / local_batch_size * epochs)
    q = local_batch_size / local_dataset_size
    gdr_steps = _count_dpdr_steps(total_steps, steps_dr, steps_interval)
    sgd_steps = total_steps - gdr_steps

    if gdr_steps > 0 and (sigma_perp <= 0 or sigma_alpha <= 0):
        mu_gdr_0 = np.inf
    elif gdr_steps > 0:
        mu_gdr_0 = math.sqrt(sigma_perp ** -2 + sigma_alpha ** -2)
    else:
        mu_gdr_0 = 0.0

    if sgd_steps > 0 and sigma_g <= 0:
        mu_sgd_0 = np.inf
    elif sgd_steps > 0:
        mu_sgd_0 = 1.0 / sigma_g
    else:
        mu_sgd_0 = 0.0

    total_mu_sq = q ** 2 * (
        gdr_steps * _poisson_gdp_term(mu_gdr_0)
        + sgd_steps * _poisson_gdp_term(mu_sgd_0)
    )
    mu_total = np.inf if math.isinf(total_mu_sq) else math.sqrt(total_mu_sq)

    if not return_details:
        return mu_total
    return {
        "mu": mu_total,
        "q": q,
        "total_steps": total_steps,
        "gdr_steps": gdr_steps,
        "sgd_steps": sgd_steps,
        "mu_gdr_0": mu_gdr_0,
        "mu_sgd_0": mu_sgd_0,
    }


def privacy_check_gdp_dpdr(local_dataset_size, local_batch_size, epochs,
                           epsilon_budget, delta_budget,
                           sigma_perp, sigma_alpha, sigma_g,
                           steps_dr=None, steps_interval=None):
    details = compute_dpdr_gdp_mu(
        local_dataset_size=local_dataset_size,
        local_batch_size=local_batch_size,
        epochs=epochs,
        sigma_perp=sigma_perp,
        sigma_alpha=sigma_alpha,
        sigma_g=sigma_g,
        steps_dr=steps_dr,
        steps_interval=steps_interval,
        return_details=True,
    )
    eps_real = gdp_epsilon_from_mu(details["mu"], delta_budget)
    print(
        "GDP privacy epsilon: %.6f (mu=%.6f, q=%.6f, total_steps=%d, "
        "gdr_steps=%d, sgd_steps=%d)"
        % (
            eps_real,
            details["mu"],
            details["q"],
            details["total_steps"],
            details["gdr_steps"],
            details["sgd_steps"],
        )
    )
    return eps_real <= epsilon_budget


def compute_dpsgd_gdp_mu(local_dataset_size, local_batch_size, epochs, sigma_g):
    if local_dataset_size <= 0 or local_batch_size <= 0:
        raise ValueError("dataset size and batch size must be positive.")
    total_steps = int(local_dataset_size / local_batch_size * epochs)
    if total_steps <= 0:
        return 0.0
    if sigma_g <= 0:
        return np.inf
    q = local_batch_size / local_dataset_size
    mu0 = 1.0 / sigma_g
    total_mu_sq = q ** 2 * total_steps * _poisson_gdp_term(mu0)
    return np.inf if math.isinf(total_mu_sq) else math.sqrt(total_mu_sq)


def privacy_check_gdp_dpsgd(local_dataset_size, local_batch_size, epochs,
                            epsilon_budget, delta_budget, sigma_g):
    total_steps = int(local_dataset_size / local_batch_size * epochs)
    q = local_batch_size / local_dataset_size
    mu = compute_dpsgd_gdp_mu(
        local_dataset_size=local_dataset_size,
        local_batch_size=local_batch_size,
        epochs=epochs,
        sigma_g=sigma_g,
    )
    eps_real = gdp_epsilon_from_mu(mu, delta_budget)
    print(
        "DPSGD GDP privacy epsilon: %.6f (mu=%.6f, q=%.6f, total_steps=%d)"
        % (eps_real, mu, q, total_steps)
    )
    return eps_real <= epsilon_budget


def privacy_check(local_dataset_size, local_batch_size, epochs, epsilon_budget, delta_budget, noise_multiplier):
    # from opacus.accountants.analysis import rdp
    T = int(local_dataset_size / local_batch_size * epochs)
    q = local_batch_size / local_dataset_size
    flag = False
    rdps = 0
    orders = [1 + x / 10.0 for x in range(1, 100)] + list(range(11, 64))+ [128, 256, 512]
    # orders = [1 + x / 10.0 for x in range(1, 100)] + list(range(2, 64)) + [128]

    for n in noise_multiplier:
        if n == 0.:
            pass
        else:
            rdps += compute_rdp(q=q, noise_multiplier=n, steps=T, orders=orders)
    if sum(noise_multiplier) == 0:
        return True
    else:
        eps_real, order = compute_eps(orders, rdps, delta_budget)
        if eps_real <= epsilon_budget or sum(noise_multiplier) == 0:
            flag = True
        print(eps_real)
        return flag

def compute_rdp(q, noise_multiplier, steps, orders):
  """Computes RDP of the Sampled Gaussian Mechanism.
  Args:
    q: The sampling rate.
    noise_multiplier: The ratio of the standard deviation of the Gaussian noise    STD标准差，敏感度应该包含在这里面了
      to the l2-sensitivity of the function to which it is added.
    steps: The number of steps.
    orders: An array (or a scalar) of RDP orders.
  Returns:
    The RDPs at all orders. Can be `np.inf`.
  """
  if np.isscalar(orders):
    rdp = _compute_rdp(q, noise_multiplier, orders)
  else:
    rdp = np.array(
        [ _compute_rdp(q, noise_multiplier, order) for order in orders])

  return rdp * steps

def _compute_log_a_for_int_alpha(q, sigma, alpha):
    assert isinstance(alpha, int)
    rdp = -np.inf

    for i in range(alpha + 1):
        log_b = (
                math.log(special.binom(alpha, i))
                + i * math.log(q)
                + (alpha - i) * math.log(1 - q)
                + (i * i - i) / (2 * (sigma ** 2))
        )


        a, b = min(rdp, log_b), max(rdp, log_b)
        if a == -np.inf:  # adding 0
            rdp = b
        else:
            rdp = math.log(math.exp(
                a - b) + 1) + b

    rdp = float(rdp) / (alpha - 1)
    return rdp


def _log_add(logx: float, logy: float) -> float:
    r"""Adds two numbers in the log space.

    Args:
        logx: First term in log space.
        logy: Second term in log space.

    Returns:
        Sum of numbers in log space.
    """
    a, b = min(logx, logy), max(logx, logy)
    if a == -np.inf:
        return b
    return math.log1p(math.exp(a - b)) + b


def _log_sub(logx: float, logy: float) -> float:
    r"""Subtracts two numbers in the log space.

    Args:
        logx: First term in log space. Expected to be greater than the second term.
        logy: First term in log space. Expected to be less than the first term.

    Returns:
        Difference of numbers in log space.

    Raises:
        ValueError
            If the result is negative.
    """
    if logx < logy:
        raise ValueError("The result of subtraction must be non-negative.")
    if logy == -np.inf:  # subtracting 0
        return logx
    if logx == logy:
        return -np.inf  # 0 is represented as -np.inf in the log space.

    try:
        return math.log(math.expm1(logx - logy)) + logy
    except OverflowError:
        return logx

def _log_erfc(x: float) -> float:
    r"""Computes :math:`log(erfc(x))` with high accuracy for large ``x``.

    Helper function used in computation of :math:`log(A_\alpha)`
    for a fractional alpha.

    Args:
        x: The input to the function

    Returns:
        :math:`log(erfc(x))`
    """
    return math.log(2) + special.log_ndtr(-x * 2 ** 0.5)

def _compute_log_a_for_frac_alpha(q: float, sigma: float, alpha: float) -> float:
    r"""Computes :math:`log(A_\alpha)` for fractional ``alpha``.

    Notes:
        Note that
        :math:`A_\alpha` is real valued function of ``alpha`` and ``q``,
        and that 0 < ``q`` < 1.

        Refer to Section 3.3 of https://arxiv.org/pdf/1908.10530.pdf for details.

    Args:
        q: Sampling rate of SGM.
        sigma: The standard deviation of the additive Gaussian noise.
        alpha: The order at which RDP is computed.

    Returns:
        :math:`log(A_\alpha)` as defined in Section 3.3 of
        https://arxiv.org/pdf/1908.10530.pdf.
    """
    # The two parts of A_alpha, integrals over (-inf,z0] and [z0, +inf), are
    # initialized to 0 in the log space:
    log_a0, log_a1 = -np.inf, -np.inf
    i = 0

    z0 = sigma ** 2 * math.log(1 / q - 1) + 0.5

    while True:  # do ... until loop
        coef = special.binom(alpha, i)
        log_coef = math.log(abs(coef))
        j = alpha - i

        log_t0 = log_coef + i * math.log(q) + j * math.log(1 - q)
        log_t1 = log_coef + j * math.log(q) + i * math.log(1 - q)

        log_e0 = math.log(0.5) + _log_erfc((i - z0) / (math.sqrt(2) * sigma))
        log_e1 = math.log(0.5) + _log_erfc((z0 - j) / (math.sqrt(2) * sigma))

        log_s0 = log_t0 + (i * i - i) / (2 * (sigma ** 2)) + log_e0
        log_s1 = log_t1 + (j * j - j) / (2 * (sigma ** 2)) + log_e1

        if coef > 0:
            log_a0 = _log_add(log_a0, log_s0)
            log_a1 = _log_add(log_a1, log_s1)
        else:
            log_a0 = _log_sub(log_a0, log_s0)
            log_a1 = _log_sub(log_a1, log_s1)

        i += 1
        if max(log_s0, log_s1) < -30:
            break

    return _log_add(log_a0, log_a1) / (alpha - 1)

def _compute_rdp(q, sigma, alpha):
    """Compute RDP of the Sampled Gaussian mechanism at order alpha.
    Args:
      q: The sampling rate.
      sigma: The std of the additive Gaussian noise.
      alpha: The order at which RDP is computed.
    Returns:
      RDP at alpha, can be np.inf.

      q==1时的公式可参考：[renyi differential privacy,2017,Proposition 7]
      0<q<1时，有以下两个公式：
      可以参考[Renyi Differential Privacy of the Sampled Gaussian Mechanism ,2019,3.3]，这篇文章中包括alpha为浮点数的计算
      公式2更为简洁的表达在[User-Level Privacy-Preserving Federated Learning: Analysis and Performance Optimization,2021,3.2和3.3]
    """
    if q == 0:
        return 0

    # no privacy
    if sigma == 0:
        return np.inf

    if q == 1.:
        return alpha  / (2 * sigma ** 2)

    if np.isinf(alpha):
        return np.inf

    if float(alpha).is_integer():
        return _compute_log_a_for_int_alpha(q, sigma, int(alpha))
    else:
        return _compute_log_a_for_frac_alpha(q, sigma, alpha)

def compute_rdp_randomized_response(p,steps,orders):

    if np.isscalar(orders):
        rdp = _compute_rdp_randomized_response(p, orders)
    else:
        rdp = np.array([_compute_rdp_randomized_response(p, order) for order in orders])

    return rdp * steps

def _compute_rdp_randomized_response(p,alpha):
    from decimal import Decimal
    a=Decimal(p**alpha)
    b=Decimal((1-p)**(1-alpha))
    c=Decimal(a*b)
    item1=float((p**alpha)*((1-p)**(1-alpha)))
    item2=float(((1-p)**alpha)*(p**(1-alpha)))
    rdp=float(math.log(item1+item2))/ (alpha-1)
    return rdp


def compute_noise_multiplier(local_dataset_size, local_batch_size, T, epsilon, delta):
    if epsilon >= 10e3:
        return 0
    q = local_batch_size / local_dataset_size
    from opacus.accountants.utils import get_noise_multiplier
    # try:
    #     nm = get_noise_multiplier(target_epsilon=epsilon, target_delta=delta,sample_rate=q,epochs=T,accountant='prv', epsilon_tolerance=1e-6)
    # except:
    #     nm = get_noise_multiplier(target_epsilon=epsilon, target_delta=delta,sample_rate=q,epochs=T,accountant='rdp', epsilon_tolerance=1e-6)
    nm = 10 * q * math.sqrt(T * (-math.log10(delta))) / epsilon
    return nm

def exp_topk(idx_topk, topk_num, epsilon):
    if torch is None:
        raise ImportError("exp_topk requires torch to be installed.")
    # 计算R中每个回复的分数
    d = len(idx_topk)
    sensitivity = d-1
    # R = np.argsort(np.abs(x))
    scores = [d-i for i in range(d)]
    # 根据分数计算每个回复的输出概率
    try:
        probabilities = [np.exp(epsilon * score / (2 * sensitivity)) for score in scores]
    except:
        print(idx_topk)
    probabilities = torch.tensor(probabilities / np.linalg.norm(probabilities, ord=1), dtype=torch.float64)

    # 根据概率分布选择回复结果
    idx_exp = torch.multinomial(probabilities, topk_num).to('cuda')
    # res = idx_topk.gather(0, idx_exp)
    return idx_exp


def compute_eps(orders, rdp, delta):
    """Compute epsilon given a list of RDP values and target delta.
    Args:
        orders: An array (or a scalar) of orders.
        rdp: A list (or a scalar) of RDP guarantees.
        delta: The target delta.
    Returns:
        Pair of (eps, optimal_order).
    Raises:
        ValueError: If input is malformed.
    """
    orders_vec = np.atleast_1d(orders)
    rdp_vec = np.atleast_1d(rdp)

    if delta <= 0:
        raise ValueError("Privacy failure probability bound delta must be >0.")
    if len(orders_vec) != len(rdp_vec):
        raise ValueError("Input lists must have the same length.")

    eps_vec = []
    for (a, r) in zip(orders_vec, rdp_vec):
        if a < 1:
            raise ValueError("Renyi divergence order must be >=1.")
        if r < 0:
            raise ValueError("Renyi divergence must be >=0.")

        if delta**2 + math.expm1(-r) >= 0:
            eps = 0
        elif a > 1.01:
            eps = ( r - (np.log(delta) + np.log(a)) / (a - 1) + np.log((a - 1) / a))
        else:
            eps = np.inf
        eps_vec.append(eps)

    idx_opt = np.argmin(eps_vec)
    return max(0, eps_vec[idx_opt]), orders_vec[idx_opt]

if __name__ == '__main__':
    batch= 4096
    datasets = 50000 #73000 SVHN #50000 cifar10 #60000 MNIST
    q=batch/datasets
    epochs=20
    steps=int(epochs*datasets/batch)


    # for dpdr MNIST
    # noise_multiplier1=0.81 #0.6 eps=8 #0.81 eps=3 #1.47 eps=1
    # noise_multiplier2=2.0 #0.8 eps=8 #2.0 eps=3 #3.5 eps=1
    # for dpsgd
    # noise_multiplier1=0.803 #eps=8 0.59 #eps=3 0.803 #eps=1 1.4
    # noise_multiplier2=0.0 #eps=8 #eps=3 0.0


    # for dpdr CIFAR10 batchsize=256
    # noise_multiplier1=0.84 #0.61 eps=8 #0.84 eps=3 #1.57 eps=1
    # noise_multiplier2=3.0 #1.0 eps=8 #3.0 eps=3 #4.0 eps=1
    # for dpsgd
    # noise_multiplier1=0.835 #0.605 eps=1 #0.835 eps=3 #1.5 eps=1
    # noise_multiplier2=0.0 

    # dpsgd eps=3 different batch size
    # noise_multiplier1=2.14 # b=64 0.666; b=256 0.84; b=1024 1.23; b=4096 2.14
    # noise_multiplier2=0.0 
    # for dpdr eps=3
    noise_multiplier1=2.2 # b=64 0.667; b=256 0.84; b=1024 1.24; b=4096 2.2
    noise_multiplier2=8.0 # b=64 2.0; b=256 3; b=1024 6; b=4096 8

    #SVHN
    #for sgd
    # noise_multiplier1=0.695 #0.527 eps=8 #0.695 eps=3  #1.075 eps=1
    # noise_multiplier2=0.0 #0.0
    #for dpdr
    # noise_multiplier1=0.696 #0.531 eps=8 #0.696 eps=3 #1.08 eps=1
    # noise_multiplier2=2.0 #0.8 eps=8 #2.0 eps=3 #3.5 eps=1

    #test
    # noise_multiplier1 = 1.8
    # noise_multiplier2=8.0 

    ORDERS = [1 + x / 10.0 for x in range(1, 100)] + list(range(2, 64)) + [128, 256, 512]
    # flag = privacy_check( datasets, 256, epochs, 3.0, 1e-5, [noise_multiplier1, noise_multiplier2])
    # print(flag)
    rdp1 = compute_rdp(q, noise_multiplier1, steps, ORDERS)
    rdp2 = compute_rdp(q, noise_multiplier2, steps, ORDERS)
    # rdp2=0
    dp1,order=compute_eps(ORDERS,rdp1,1e-5)
    print("dp1:",dp1)
    # print("order:",order)
    dp2,order=compute_eps(ORDERS,rdp2,1e-5)
    print("dp2:",dp2)
    # print("order:",order)
    dp,order=compute_eps(ORDERS,rdp1+rdp2,1e-5)
    # print("rdp:",rdp)
    print("dp:",dp)
    print("order:",order)
    # print("rdp:",rdp)
    # print("dp:",dp)
    # print("order:",order)

    # steps = 0
    # for e in range(1, epochs+1):
    #     steps = int(e*datasets/batch)
    #     rdp1 = compute_rdp(q, noise_multiplier1, steps, ORDERS)
    #     # rdp2 = compute_rdp(q, noise_multiplier2, steps, ORDERS)
    #     rdp2 = 0
    #     dp,order=compute_eps(ORDERS,rdp1+rdp2,1e-5)
    #     print("epochs, dp:",dp)
    #     # print("order:",order)
