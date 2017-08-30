#!usr/bin/env python
#-*- coding:utf-8 -*-
"""
@author: Jeff Zhang
@date:   2017-08-30
"""


import autograd.numpy as np
from autograd.scipy.misc import logsumexp
from autograd.convenience_wrappers import value_and_grad as vgrad
from functools import partial





def EM(init_params, data, callback=None):
    def EM_update(params):
        natural_params = list(map(np.log, params))
        loglike, E_stats = vgrad(log_partition_function)(natural_params, data)  # E step
        if callback: callback(loglike, params)
        return list(map(normalize, E_stats))                                    # M step

    def fixed_point(f, x0):
        x1 = f(x0)
        while different(x0, x1):
            x0, x1 = x1, f(x1)
        return x1

    def different(params1, params2):
        allclose = partial(np.allclose, atol=1e-3, rtol=1e-3)
        return not all(map(allclose, params1, params2))

    return fixed_point(EM_update, init_params)



def normalize(a):
    def replace_zeros(a):
        return np.where(a > 0., a, 1.)
    return a / replace_zeros(a.sum(-1, keepdims=True))


def log_partition_function(natural_params, data):
    if isinstance(data, list):
        return sum(map(partial(log_partition_function, natural_params), data))

    log_pi, log_A, log_B = natural_params

    log_alpha = log_pi
    for y in data:
        log_alpha = logsumexp(log_alpha[:,None] + log_A, axis=0) + log_B[:,y]

    return logsumexp(log_alpha)