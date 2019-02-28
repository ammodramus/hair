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

import lm2 as likm
import scipy.stats as st

def log_prior_A(params, *args):
    f, N1 = params
    if f < 0 or f > 1:
        return -np.inf
    if N1 <= 1 or N1 >= 500:
        return -np.inf
    return 0.0

def log_post_A(params, *args):
    log_prior = log_prior_A(params)
    if not np.isfinite(log_prior):
        return log_prior
    val = log_prior + -1.0*likelihood_model_A_two_bots(params, *args)
    return val

#################
# MODEL B
#################

def log_likelihood_model_B(params, hair_freqs, blood_freqs, cheek_freqs):
    f, N1, N2 = params
    indiv_log_likes = []
    for hair_i, cheek_i, blood_i in zip(hair_freqs, cheek_freqs, blood_freqs):
        s = f*blood_i + (1-f)*cheek_i
        indiv_hair_logprobs = []
        for k in range(0,int(N1)+1):
            log_binom_prob = st.binom.logpmf(k, N1, s)
            p_B1 = float(k)/N1
            logprob = 0.0
            for hair_ij in hair_i:
                val = np.log(likm.convolve_binom_and_beta(hair_ij, N2, p_B1))
                if np.isnan(val) or val == np.inf:
                    with np.errstate(all='ignore'):
                        val = np.log(likm.convolve_binom_and_beta_sum(hair_ij, N2, p_B1))
                logprob += val
            logprob += log_binom_prob
            indiv_hair_logprobs.append(logprob)
        val2 = scipy.special.logsumexp(indiv_hair_logprobs)
        indiv_log_likes.append(val2)
    return scipy.special.logsumexp(indiv_log_likes)

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


def analysis_A():
    dat = pd.read_excel('som_hair_freqs.xlsx', 0)

    with np.errstate(all='ignore'):
        n_walkers = 50
        args = [dat.hair.values, dat.blood.values, dat.cheek.values]
        samp = emcee.EnsembleSampler(n_walkers, 2, log_post_A, args=args, threads=8)
        p0_f = npr.uniform(0, 1, size=n_walkers)
        p0_N = npr.uniform(1, 500, size=n_walkers)
        p0 = np.column_stack((p0_f, p0_N))
        positions = []
        lnprobs = []
        n_iter = 1000
        for i, (chainpos, chainlnprobs, _) in enumerate(samp.sample(p0 = p0, iterations=n_iter)):
            chainpos = np.column_stack((chainlnprobs[:,np.newaxis], chainpos))
            positions.append(chainpos)
            print '# {} of {}'.format(i+1, n_iter)

        res = opt.minimize(likelihood_model_A_two_bots, np.array((0.5, 10)), method='Nelder-Mead', args=tuple(args))
        print res.x, res.fun, res.nfev, res.nit, res.status
        res = opt.minimize(likelihood_model_A_two_bots, np.array((0.5, 100)), method='Nelder-Mead', args=tuple(args))
        print res.x, res.fun, res.nfev, res.nit, res.status


    np.savetxt('positions_N_uniform.txt', np.concatenate(positions), delimiter=',')

    def log_prior_A(params, *args):
        ''' this is now log-uniform in N (and 1/N) '''
        f, N1 = params
        if f < 0 or f > 1:
            return -np.inf
        if N1 <= 1 or N1 >= 500:
            return -np.inf
        return 1/N1

    def log_post_A(params, *args):
        log_prior = log_prior_A(params)
        if not np.isfinite(log_prior):
            return log_prior
        val = log_prior + -1.0*likelihood_model_A_two_bots(params, *args)
        return val


    with np.errstate(all='ignore'):
        n_walkers = 50
        args = [dat.hair.values, dat.blood.values, dat.cheek.values]
        samp = emcee.EnsembleSampler(n_walkers, 2, log_post_A, args=args, threads=8)
        p0_f = npr.uniform(0, 1, size=n_walkers)
        p0_N = npr.uniform(1, 500, size=n_walkers)
        p0 = np.column_stack((p0_f, p0_N))
        positions = []
        lnprobs = []
        n_iter = 1000
        for i, (chainpos, chainlnprobs, _) in enumerate(samp.sample(p0 = p0, iterations=n_iter)):
            chainpos = np.column_stack((chainlnprobs[:,np.newaxis], chainpos))
            positions.append(chainpos)
            print '# {} of {}'.format(i+1, n_iter)


    np.savetxt('positions_N_log_uniform.txt', np.concatenate(positions), delimiter=',')


def log_prior_B_uniform(params, *args):
    ''' uniform in f, N1, and N2 '''
    f, N1, N2 = params
    if f < 0 or f > 1:
        return -np.inf
    if N1 <= 1 or N1 >= 500:
        return -np.inf
    if N2 <= 1 or N2 >= 500:
        return -np.inf
    return 0.0

def log_post_B_uniform(params, *args):
    log_prior = log_prior_B_uniform(params)
    if not np.isfinite(log_prior):
        return log_prior
    #val = log_prior + log_likelihood_model_B_two_bots(params, *args)
    val = log_prior + log_likelihood_model_B_one_bot(params, *args)
    #print '#', val, params
    return val


def log_prior_B_loguniform(params, *args):
    ''' loguniform in f, N1, and N2 '''
    f, N1, N2 = params
    if f < 0 or f > 1:
        return -np.inf
    if N1 <= 1 or N1 >= 500+1:
        return -np.inf
    if N2 <= 1 or N2 >= 500+1:
        return -np.inf
    return 1/N1 + 1/N2


def log_post_B_loguniform(params, *args):
    log_prior = log_prior_B_loguniform(params)
    if not np.isfinite(log_prior):
        return log_prior
    #val = log_prior + log_likelihood_model_B_two_bots(params, *args)
    val = log_prior + log_likelihood_model_B_one_bot(params, *args)
    #print '#', val, params
    return val


def analysis_B():
    allh = pd.read_excel('all_hair_freqs.xlsx', 0)
    del allh['som']
    allh['idx'] = np.concatenate(allh.groupby('individual_id').apply(lambda x: np.arange(x.shape[0])).values)
    allh.set_index(['individual_id', 'idx'], inplace=True)
    somf = pd.read_excel('indiv_cheek_hair.xlsx', 0)
    allh = allh.reset_index().merge(somf, left_on='individual_id', right_on='individual_id').set_index(['individual_id', 'idx'])
    model_args = zip(*allh.groupby(level=1).apply(lambda x: [x['hair'].values, x['cheek'].iloc[0], x['blood'].iloc[0]]).tolist())

    with np.errstate(all='ignore'):
        n_walkers = 100
        samp = emcee.EnsembleSampler(n_walkers, 3, log_post_B_loguniform, args=model_args, threads=20)
        p0_f = npr.uniform(0, 1, size=n_walkers)
        p0_N1 = npr.uniform(1, 500, size=n_walkers)
        p0_N2 = npr.uniform(1, 500, size=n_walkers)
        p0 = np.column_stack((p0_f, p0_N1, p0_N2))
        positions = []
        n_iter = 20000
        for i, (chainpos, chainlnprobs, _) in enumerate(samp.sample(p0 = p0, iterations=n_iter)):
            chainpos = np.column_stack((chainlnprobs[:,np.newaxis], chainpos))
            positions.append(chainpos)
            np.savetxt(sys.stdout, chainpos, delimiter='\t')


    print '# saving loguniform'
    np.savetxt('positions_N_log_uniform_B.txt', np.concatenate(positions), delimiter=',')

if __name__ == '__main__':
    analysis_B()
