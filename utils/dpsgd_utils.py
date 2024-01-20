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
import torch
# from opacus.accountants.utils import get_noise_multiplier
# from opacus import PrivacyEngine
np.random.seed(10)


def compute_noise_multiplier(local_dataset_size, local_batch_size, T, epsilon, delta):
    if epsilon >= 10e3:
        return 0
    q = local_batch_size / local_dataset_size
    from opacus.accountants.utils import get_noise_multiplier
    nm = get_noise_multiplier(target_epsilon=epsilon, target_delta=delta,sample_rate=q,epochs=T,accountant='prv', epsilon_tolerance=1e-6)
    # nm = 10 * q * math.sqrt(T * (-math.log10(delta))) / epsilon
    return nm

def exp_topk(idx_topk, topk_num, epsilon):
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