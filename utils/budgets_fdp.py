import numpy as np
from scipy.stats import norm

# for privacy, given eps, return delta
def theoretical_delta_fdp_dpdr(sig_perp, sig_alpha, T, q, eps):
    mu_grad = np.sqrt(1/sig_perp**2 + 1/sig_alpha**2)
    # print(mu_grad)
    c = q*np.sqrt(T)
    # print(c)
    mu_dpdr = np.sqrt(2) * c * np.sqrt(np.exp(mu_grad) * norm.cdf(1.5*np.sqrt(mu_grad))+3*norm.cdf(-0.5*np.sqrt(mu_grad))-2)
    delta = norm.cdf(-eps/mu_dpdr + mu_dpdr/2) - np.exp(eps)*norm.cdf(-eps/mu_dpdr - mu_dpdr/2)
    return delta

def theoretical_delta_fdp_dpsgd(sig_g, T, q, eps):
    mu_grad = 1/sig_g
    # print(mu_grad)
    c = q*np.sqrt(T)
    mu_dpdr = np.sqrt(2) * c * np.sqrt(np.exp(mu_grad) * norm.cdf(1.5*np.sqrt(mu_grad))+3*norm.cdf(-0.5*np.sqrt(mu_grad))-2)
    delta = norm.cdf(-eps/mu_dpdr + mu_dpdr/2) - np.exp(eps)*norm.cdf(-eps/mu_dpdr - mu_dpdr/2)
    return delta

# search eps
def binary_search_eps_dpdr(delta, param, f):
    """
    使用二分查找法求解满足 delta = f(eps) 的 eps 值
    
    参数:
    e_low: eps的下界
    e_high: eps的上界
    f: 函数d=f(e)
    tolerance: 允许的误差范围
    max_iterations: 最大迭代次数
    
    返回:
    e: 满足条件的e值
    """
    max_iterations = 1000
    tolerance =1e-9
    sig_perp, sig_alpha, T, q = param
    # 检查边界条件
    e_low = 1e-5
    e_high = 50
    
    d_low = f(sig_perp, sig_alpha, T, q, e_low)
    d_high = f(sig_perp, sig_alpha, T, q, e_high)
    
    # 检查目标值是否在范围内
    if (delta - d_low) * (delta - d_high) > 0:
        raise ValueError(f"delta {delta} 不在e[1e-5, 50]范围内, 对应delta范围[{d_low}, {d_high}]")
    
    # 二分查找
    for i in range(max_iterations):
        e_mid = (e_low + e_high) / 2
        d_mid = f(sig_perp, sig_alpha, T, q, e_mid)
        
        # 检查是否满足精度要求
        if abs(d_mid - delta) < tolerance:
            return e_mid
        
        # 调整搜索区间
        if (d_mid < delta and d_low < d_high) or (d_mid > delta and d_low > d_high):
            e_low = e_mid
            d_low = d_mid
        else:
            e_high = e_mid
            d_high = d_mid
    
    # 如果达到最大迭代次数仍未满足精度要求
    e_mid = (e_low + e_high) / 2
    print(f"达到最大迭代次数，当前误差: {abs(f(sig_perp, sig_alpha, T, q, e_mid) - delta)}")
    return e_mid

# search eps
def binary_search_eps_dpsgd(delta, param, f):
    """
    使用二分查找法求解满足 delta = f(eps) 的 eps 值
    
    参数:
    e_low: eps的下界
    e_high: eps的上界
    f: 函数d=f(e)
    tolerance: 允许的误差范围
    max_iterations: 最大迭代次数
    
    返回:
    e: 满足条件的e值
    """
    max_iterations = 1000
    tolerance =1e-9
    sig_g, T, q = param
    # 检查边界条件
    e_low = 1e-5
    e_high = 50
    
    d_low = f(sig_g, T, q, e_low)
    d_high = f(sig_g, T, q, e_high)
    
    # 检查目标值是否在范围内
    if (delta - d_low) * (delta - d_high) > 0:
        raise ValueError(f"delta {delta} 不在e[1e-5, 50]范围内, 对应delta范围[{d_low}, {d_high}]")
    
    # 二分查找
    for i in range(max_iterations):
        e_mid = (e_low + e_high) / 2
        d_mid = f(sig_g, T, q, e_mid)
        
        # 检查是否满足精度要求
        if abs(d_mid - delta) < tolerance:
            return e_mid
        
        # 调整搜索区间
        if (d_mid < delta and d_low < d_high) or (d_mid > delta and d_low > d_high):
            e_low = e_mid
            d_low = d_mid
        else:
            e_high = e_mid
            d_high = d_mid
    
    # 如果达到最大迭代次数仍未满足精度要求
    e_mid = (e_low + e_high) / 2
    print(f"达到最大迭代次数，当前误差: {abs(f(sig_g, T, q, e_mid) - delta)}")
    return e_mid

# 示例使用
if __name__ == "__main__":
    # MNIST
    sig_perp = 0.81
    sig_alpha = 2.0
    sig_g = 0.803
    eps=3

    # sig_perp = 0.6
    # sig_alpha = 0.8
    # sig_g = 0.59
    # eps = 8

    batch= 256
    datasets = 60000 #73000 SVHN #50000 cifar10 #60000 MNIST
    q=batch/datasets
    epochs=20
    T=int(epochs*datasets/batch)

    # fdp bound
    #dpdr
    res_delta = theoretical_delta_fdp_dpdr(sig_perp, sig_alpha, T, q, eps)
    print(f"when eps = {eps}, delta is {res_delta}")

    given_delta = 1e-5
    param = (sig_perp, sig_alpha, T, q)
    res_eps = binary_search_eps_dpdr(given_delta, param, theoretical_delta_fdp_dpdr)
    print(f"当 delta = {given_delta} 时, eps ≈ {res_eps}")
    print(f"验证: f({res_eps}) = {theoretical_delta_fdp_dpdr(sig_perp, sig_alpha, T, q, res_eps)}")

    #dpsgd
    res_delta = theoretical_delta_fdp_dpsgd(sig_g, T, q, eps)
    print(f"when eps = {eps}, delta is {res_delta}")

    given_delta = 1e-5
    param = (sig_g, T, q)
    res_eps = binary_search_eps_dpsgd(given_delta, param, theoretical_delta_fdp_dpsgd)
    print(f"当 delta = {given_delta} 时, eps ≈ {res_eps}")
    print(f"验证: f({res_eps}) = {theoretical_delta_fdp_dpsgd(sig_g, T, q, res_eps)}")

