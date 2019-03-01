
# coding: utf-8

from __future__ import division
import re
import sys

import pandas as pd
import numpy as np
import numpy.random as npr
from numba import jit
import scipy.stats
import scipy.special
import scipy.optimize as opt
import emcee
import scipy.stats as st

import mpmath as mp
mp.dps = 20


@jit
def convolve_binom_and_beta_sum(x, n, p):
    if p == 0.0 or p == 1.0:
        if x > 0 and x < 1:
            return 0.0
    if (p == 1.0 and x == 1.0) or (p == 0.0 and x == 0.0):
        return 1.0
    if x == 0.0:
        return st.binom.pmf(0, n, p)
    elif x == 1.0:
        return st.binom.pmf(n, n, p)

    logterms = np.zeros(n+1)
    for j in range(1, n):
        logterms[j] = scipy.stats.binom.logpmf(j, n, p) + scipy.stats.beta.logpdf(x, j, (n-j))
    logterms[0] = scipy.stats.binom.logpmf(0, n, p) + np.log(x == 0)
    logterms[-1] = scipy.stats.binom.logpmf(n, n, p) + np.log(x == 1)
    logprob = scipy.special.logsumexp(logterms)
    return np.exp(logprob)

@jit
def convolve_binom_and_beta(x, n, p):
    if p == 0.0 or p == 1.0:
        if x > 0 and x < 1:
            return 0.0
    if (p == 1.0 and x == 1.0) or (p == 0.0 and x == 0.0):
        return 1.0
    if x == 0.0:
        return st.binom.pmf(0, n, p)
    elif x == 1.0:
        return st.binom.pmf(n, n, p)
    prob = -n*(n-1)*(p-1)*p*((p-1)*(x-1))**(n-2)*scipy.special.hyp2f1(1-n, 2-n, 2, x*p/((x-1)*(p-1)))
    if np.isnan(prob):
        prob = float(-n*(n-1)*(p-1)*p*((p-1)*(x-1))**(n-2)*mp.hyp2f1(1-n, 2-n, 2, x*p/((x-1)*(p-1))))
    return prob

def log_convolve_binom_and_beta_mp(x, n, p):
    if p == 0.0 or p == 1.0:
        if x > 0 and x < 1:
            return -np.inf
    if (p == 1.0 and x == 1.0) or (p == 0.0 and x == 0.0):
        return 0.0
    if x == 0.0:
        return st.binom.logpmf(0, n, p)
    elif x == 1.0:
        return st.binom.logpmf(n, n, p)
    n = mp.mpf(n)
    p = mp.mpf(p)
    x = mp.mpf(x)
    logprob = mp.log(-n*(n-1)*(p-1)*p*((p-1)*(x-1))**(n-2)*mp.hyp2f1(1-n, 2-n, 2, x*p/((x-1)*(p-1))))
    return float(logprob)

def log_convolve_binom_and_beta(x, n, p):
    if p == 0.0 or p == 1.0:
        if x > 0 and x < 1:
            return -np.inf
    if (p == 1.0 and x == 1.0) or (p == 0.0 and x == 0.0):
        return 0.0
    if x == 0.0:
        return st.binom.logpmf(0, n, p)
    elif x == 1.0:
        return st.binom.logpmf(n, n, p)

    logprob = np.log(-n*(n-1)*(p-1)*p*((p-1)*(x-1))**(n-2)*scipy.special.hyp2f1(1-n, 2-n, 2, x*p/((x-1)*(p-1))))
    if not np.isfinite(logprob):
        return log_convolve_binom_and_beta_mp(x, n, p)
    return logprob


#################
# MODEL B
#################

#@jit
def log_likelihood_model_B(params, hair_freqs, blood_freqs, cheek_freqs):
    f, N1, N2 = params
    total_loglike = 0.0
    for hair_i, cheek_i, blood_i in zip(hair_freqs, cheek_freqs, blood_freqs):
        s = f*blood_i + (1-f)*cheek_i
        freq_n1_logprobs = []
        for k in range(0,int(N1)+1):
            log_binom_prob = st.binom.logpmf(k, N1, s)
            freq_logprob = log_binom_prob   # the logprobability for this outcome of the first bottleneck
            p_B1 = float(k)/N1
            for hair_ij in hair_i:
                val = log_convolve_binom_and_beta(hair_ij, N2, p_B1)
                if val == -np.inf and p_B1 > 0 and p_B1 < 1:
                    print 'bad convolution:', hair_ij, N2, p_B1, '--', val
                freq_logprob += val
            freq_n1_logprobs.append(freq_logprob)
        indiv_loglike = scipy.special.logsumexp(freq_n1_logprobs)
        total_loglike += indiv_loglike
    return total_loglike


def log_likelihood_model_B_array(params, hair_freqs, blood_freqs, cheek_freqs):
    tup_params = tuple(params)
    return log_likelihood_model_B(tup_params, hair_freqs, blood_freqs, cheek_freqs)


def log_likelihood_model_B_two_bots(params, *args):
    f, N1, N2 = params
    if f < 0 or f > 1:
        return -np.inf
    if N1 < 1:
        return -np.inf
    if N2 < 1:
        return -np.inf
    N1a = int(np.floor(N1))
    N1b = int(np.ceil(N1))
    N2a = int(np.floor(N2))
    N2b = int(np.ceil(N2))
    
    frac_b_N1 = (N1-N1a)/(N1b-N1a)
    frac_b_N2 = (N2-N2a)/(N2b-N2a)

    if np.isnan(frac_b_N1):
        frac_b_N1 = 1.0
    if np.isnan(frac_b_N2):
        frac_b_N2 = 1.0

    loglike_aa = log_likelihood_model_B_array((f, N1a, N2a), *args)
    loglike_ab = log_likelihood_model_B_array((f, N1a, N2b), *args)
    loglike_ba = log_likelihood_model_B_array((f, N1b, N2a), *args)
    loglike_bb = log_likelihood_model_B_array((f, N1b, N2b), *args)
    
    weight_aa = (1-frac_b_N1)*(1-frac_b_N2)
    weight_ab = (1-frac_b_N1)*(frac_b_N2)
    weight_ba = (frac_b_N1)*(1-frac_b_N2)
    weight_bb = (frac_b_N1)*(frac_b_N2)
    weights = [weight_aa, weight_ab, weight_ba, weight_bb]
    loglikes = [loglike_aa, loglike_ab, loglike_ba, loglike_bb]
    val = scipy.special.logsumexp(loglikes, b=weights)
    return val


def log_likelihood_model_B_one_bot(params, *args):
    f, N1, N2 = params
    if f < 0 or f > 1:
        return -np.inf
    if N1 < 2:
        return -np.inf
    if N2 < 2:
        return -np.inf
    N1a = int(np.floor(N1))
    N2a = int(np.floor(N2))
    
    loglike_aa = log_likelihood_model_B_array((f, N1a, N2a), *args)
    return loglike_aa


