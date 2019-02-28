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

#################
# MODEL B
#################


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
    #val = log_prior + likm.log_likelihood_model_B_two_bots(params, *args)
    val = log_prior + likm.log_likelihood_model_B_one_bot(params, *args)
    return val


def log_post_B_uniform_two_bots(params, *args):
    log_prior = log_prior_B_uniform(params)
    if not np.isfinite(log_prior):
        return log_prior
    val = log_prior + likm.log_likelihood_model_B_two_bots(params, *args)
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
    val = log_prior + likm.log_likelihood_model_B_one_bot(params, *args)
    return val


def analysis_B(do_logunif):
    unif_str = 'logunif' if do_logunif else 'unif'
    allh = pd.read_excel('all_hair_freqs.xlsx', 0)
    del allh['som']
    allh['idx'] = np.concatenate(allh.groupby('individual_id').apply(lambda x: np.arange(x.shape[0])).values)
    allh.set_index(['individual_id', 'idx'], inplace=True)
    somf = pd.read_excel('indiv_cheek_hair.xlsx', 0)
    allh = allh.reset_index().merge(somf, left_on='individual_id', right_on='individual_id').set_index(['individual_id', 'idx'])
    model_args = zip(*allh.groupby(level=1).apply(lambda x: [x['hair'].values, x['cheek'].iloc[0], x['blood'].iloc[0]]).tolist())


    with np.errstate(all='ignore'):
        n_walkers = 100
        if do_logunif:
            pass
        else:
            samp = emcee.EnsembleSampler(n_walkers, 3, log_post_B_uniform, args=model_args, threads=20)
        p0_f = npr.uniform(0, 1, size=n_walkers)
        p0_N1 = npr.uniform(1, 500+1, size=n_walkers)
        p0_N2 = npr.uniform(1, 500+1, size=n_walkers)
        p0 = np.column_stack((p0_f, p0_N1, p0_N2))
        positions = []
        n_iter = 20000
        for i, (chainpos, chainlnprobs, _) in enumerate(samp.sample(p0 = p0, iterations=n_iter)):
            chainpos = np.column_stack((chainlnprobs[:,np.newaxis], chainpos))
            np.savetxt(sys.stdout, chainpos, delimiter='\t')
            positions.append(chainpos)

    print '# saving {}'.format(unif_str)
    np.savetxt('positions_N_{}_B.txt'.format(unif_str), np.concatenate(positions), delimiter=',')

def max_like_B():
    allh = pd.read_excel('all_hair_freqs.xlsx', 0)
    del allh['som']
    allh['idx'] = np.concatenate(allh.groupby('individual_id').apply(lambda x: np.arange(x.shape[0])).values)
    allh.set_index(['individual_id', 'idx'], inplace=True)
    somf = pd.read_excel('indiv_cheek_hair.xlsx', 0)
    allh = allh.reset_index().merge(somf, left_on='individual_id', right_on='individual_id').set_index(['individual_id', 'idx'])
    model_args = zip(*allh.groupby(level=1).apply(lambda x: [x['hair'].values, x['cheek'].iloc[0], x['blood'].iloc[0]]).tolist())

    def target(x, args):
        val = -1*log_post_B_uniform_two_bots(x, *args)
        print x, -val
        return val



    with np.errstate(all='ignore'):
        p0_f = npr.uniform(0, 1)
        p0_N1 = npr.uniform(1, 500+1)
        p0_N2 = npr.uniform(1, 500+1)
        p0 = np.array((p0_f, p0_N1, p0_N2))
        res = opt.minimize(target, x0=p0, args=model_args, method='Nelder-Mead')
        print res

if __name__ == '__main__':
    #analysis_B(do_logunif=False)
    max_like_B()
