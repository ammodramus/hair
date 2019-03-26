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
    if N1 <= 1 or N1 >= 501:
        return -np.inf
    if N2 <= 1 or N2 >= 501:
        return -np.inf
    return 0.0

def log_post_B_uniform(params, *args):
    log_prior = log_prior_B_uniform(params)
    if not np.isfinite(log_prior):
        return log_prior
    val = log_prior + likm.log_likelihood_model_B_one_bot(params, *args)
    return val


def log_post_B_uniform_two_bots(params, *args):
    log_prior = log_prior_B_uniform(params)
    if not np.isfinite(log_prior):
        return log_prior
    val = log_prior + likm.log_likelihood_model_B_two_bots(params, *args)
    return val


def log_prior_B_loguniform_two_bots(params, *args):
    ''' loguniform in f, N1, and N2 '''
    f, N1, N2 = params
    if f < 0 or f > 1:
        return -np.inf
    if N1 <= 2 or N1 >= 500+1:
        return -np.inf
    if N2 <= 2 or N2 >= 500+1:
        return -np.inf
    return 1/N1 + 1/N2


def log_prior_B_loguniform_one_bot(params, *args):
    ''' loguniform in f, N1, and N2 '''
    f = params[0]
    N1, N2 = params[1:].astype(int).abs()
    if f < 0 or f > 1:
        return -np.inf
    if N1 <= 2 or N1 >= 500+1:
        return -np.inf
    if N2 <= 2 or N2 >= 500+1:
        return -np.inf
    return np.log(1.0/N1 + 1.0/N2)


def log_post_B_loguniform(params, *args):
    log_prior = log_prior_B_loguniform_two_bots(params)

    if not np.isfinite(log_prior):
        return log_prior
    val = log_prior + likm.log_likelihood_model_B_one_bot(params, *args)
    return val


def analysis_B(do_logunif, sims):
    unif_str = 'logunif' if do_logunif else 'unif'
    if sims:
        allh = pd.read_csv('sim_hairs_data_f0p6_n1_78_n2_20.tsv', sep='\t')
        allh.set_index(['individual_id', 'idx'], inplace=True)
    else:
        allh = pd.read_csv('fixed_hair_data_26032019.tsv', sep='\t')
        allh.set_index(['individual_id', 'position', 'idx'], inplace=True)
    model_args = zip(*allh.groupby(level=[0,1]).apply(lambda x: [x['hair'].values, x['blood'].iloc[0], x['cheek'].iloc[0]]).tolist())


    with np.errstate(all='ignore'):
        n_walkers = 100
        if do_logunif:
            samp = emcee.EnsembleSampler(n_walkers, 3, log_post_B_loguniform, args=model_args, threads=NUM_THREADS)
        else:
            samp = emcee.EnsembleSampler(n_walkers, 3, log_post_B_uniform, args=model_args, threads=NUM_THREADS)
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



def analysis_sims(do_logunif):
    unif_str = 'logunif' if do_logunif else 'unif'

    allh = pd.read_csv('sim_hairs_data_f0p6_n1_78_n2_20.tsv', sep='\t')
    allh.set_index(['individual_id', 'idx'], inplace=True)
    model_args = zip(*allh.groupby(level=0).apply(lambda x: [x['hair'].values, x['blood'].iloc[0], x['cheek'].iloc[0]]).tolist())


    with np.errstate(all='ignore'):
        n_walkers = 100
        if do_logunif:
            samp = emcee.EnsembleSampler(n_walkers, 3, log_post_B_loguniform, args=model_args, threads=NUM_THREADS)
        else:
            samp = emcee.EnsembleSampler(n_walkers, 3, log_post_B_uniform, args=model_args, threads=NUM_THREADS)
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
    np.savetxt('positions_sims_N_{}_B.txt'.format(unif_str), np.concatenate(positions), delimiter=',')



def max_like_B(do_sims):
    if do_sims:
        allh = pd.read_csv('sim_hairs_data_f0p6_n1_78_n2_20.tsv', sep='\t')
        allh.set_index(['individual_id', 'idx'], inplace=True)

    else:
        allh = pd.read_csv('fixed_hair_data_26032019.tsv', sep='\t')
        allh.set_index(['individual_id', 'position', 'idx'], inplace=True)

    model_args = zip(*allh.groupby(level=[0,1]).apply(lambda x: [x['hair'].values, x['blood'].iloc[0], x['cheek'].iloc[0]]).tolist())



    def target(x, args):
        val = -1*likm.log_likelihood_model_B_two_bots(x, *args)
        print x, -val
        return val

    if do_sims:
        true_params = np.array((0.6, 78, 20))
        true_loglike = likm.log_likelihood_model_B_two_bots(true_params, *model_args)
        print 'true params:', true_params
        print 'true loglike:', true_loglike

    with np.errstate(all='ignore'):
        p0_f = npr.uniform(0, 1)
        p0_N1 = npr.uniform(2, 500+1)
        p0_N2 = npr.uniform(2, 500+1)
        p0 = np.array((p0_f, p0_N1, p0_N2))
        res = opt.minimize(target, x0=p0, args=model_args, method='Nelder-Mead')
        print res

    def target_N1_inf(x, args):
        target_val = -1*likm.log_likelihood_model_B_two_bots_N1_inf(x, *model_args)
        print x, -target_val
        return target_val

    print '------------------'
    print 'N1 large:'
    with np.errstate(all='ignore'):
        p0_f_N1_inf = npr.uniform(0, 1)
        p0_N1_N1_inf = npr.uniform(2, 500+1)
        p0 = np.array((p0_f_N1_inf, p0_N1_N1_inf))
        res = opt.minimize(target_N1_inf, x0=p0, args=model_args, method='Nelder-Mead')
        print res


    # this doesn't work because there are both zero and non-zero frequencies.
    # The beta distribution can't handle zero.

    #def target_N2_inf(x, args):
    #    target_val = -1*likm.log_likelihood_model_B_two_bots_N2_inf(x, *model_args)
    #    print x, -target_val
    #    return target_val

    #print '------------------'
    #print 'N2 large:'
    #with np.errstate(all='ignore'):
    #    p0_f_N2_inf = npr.uniform(0, 1)
    #    p0_N1_N2_inf = npr.uniform(2, 500+1)
    #    p0 = np.array((p0_f_N2_inf, p0_N1_N2_inf))
    #    res = opt.minimize(target_N2_inf, x0=p0, args=model_args, method='Nelder-Mead')
    #    print res
    #'''


NUM_THREADS = None


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(
            description='perform MCMC for hairs',
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--log-unif', help='log-uniform priors', action='store_true')
    parser.add_argument('--max-like', help='maximum-likelihood estimation', action='store_true')
    parser.add_argument('--sims', action='store_true')
    parser.add_argument('--num-threads', type=int, default=20)
    args = parser.parse_args()

    NUM_THREADS = args.num_threads

    if args.max_like:
        max_like_B(args.sims)
    else:
        analysis_B(do_logunif=args.log_unif, sims=args.sims)
