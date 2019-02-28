
# coding: utf-8

# In[1]:

get_ipython().magic(u'matplotlib inline')
from __future__ import division
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy.random as npr
import re
from numba import jit
import scipy.stats
import scipy.special


# In[2]:

def hpd(trace, mass_frac):
    d = np.sort(np.copy(trace))
    n = len(trace)
    n_samples = np.floor(mass_frac * n).astype(int)
    int_width = d[n_samples:] - d[:n-n_samples]
    min_int = np.argmin(int_width)
    return np.array([d[min_int], d[min_int+n_samples]])


# In[170]:

dat = pd.read_excel('som_hair_freqs.xlsx', 0)


# In[171]:

@jit
def likelihood_model_A(params, hair, blood, cheek):
    f, N1 = params
    N1 = int(N1)
    total_logprob = 0
    n_samples = hair.shape[0]
    for i in range(n_samples):
        h = hair[i]
        c = cheek[i]
        b = blood[i]
        s = f*b + (1-f)*c
        prob = 0.0
        logterms = np.zeros(N1+1)
        for j in range(1, N1):
            logterms[j] = scipy.stats.binom.logpmf(j, N1, s) + scipy.stats.beta.logpdf(h, j, (N1-j))
        logterms[0] = scipy.stats.binom.logpmf(0, N1, s) + np.log(h == 0)
        logterms[-1] = scipy.stats.binom.logpmf(N1, N1, s) + np.log(h == 1)
        logprob = scipy.special.logsumexp(logterms)
        total_logprob += logprob
    return total_logprob


@jit
def likelihood_model_A_hyp2f1(params, hair, blood, cheek):
    f, N1 = params
    N1 = int(N1)
    total_logprob = 0
    n_samples = hair.shape[0]
    for i in range(n_samples):
        h = hair[i]
        c = cheek[i]
        b = blood[i]
        s = f*b + (1-f)*c
        prob = -N1*(N1-1)*(s-1)*((s-1)*(h-1))**(N1-2)*s*scipy.special.hyp2f1(1-N1, 2-N1, 2, h*s/((h-1)*(s-1)))
        logprob = np.log(prob)
        total_logprob += logprob
    return total_logprob


# In[172]:

likelihood_model_A_hyp2f1((0.5, 20), dat.hair.values, dat.blood.values, dat.cheek.values)


# In[173]:

@jit
def likelihood_model_A_two_bots(params, hair, blood, cheek):
    f, N1 = params
    if f < 0 or f > 1:
        return np.inf
    if N1 < 1:
        return np.inf
    N1a = np.floor(N1)
    N1b = np.ceil(N1)
    loglike_a = likelihood_model_A_hyp2f1((f, N1a), hair, blood, cheek)
    if not np.isfinite(loglike_a):
        loglike_a = likelihood_model_A((f, N1a), hair, blood, cheek)
    loglike_b = likelihood_model_A_hyp2f1((f, N1b), hair, blood, cheek)
    if not np.isfinite(loglike_b):
        loglike_b = likelihood_model_A((f, N1b), hair, blood, cheek)
    frac_b = N1-N1a
    val = -scipy.special.logsumexp((loglike_a, loglike_b), b=(1-frac_b, frac_b))
    return val


# In[174]:

import scipy.optimize as opt


# In[175]:

opt.minimize(likelihood_model_A_two_bots, np.array((0.5, 10)), args=(dat.hair.values, dat.blood.values, dat.cheek.values), method='Nelder-Mead')


# In[176]:

opt.minimize(likelihood_model_A_two_bots, np.array((0.5, 10)), args=(dat.hair.values, dat.blood.values, dat.cheek.values, 0.5), method='Nelder-Mead')


# In[ ]:

opt.minimize(likelihood_model_A_two_bots, np.array((0.5, 10)), args=(dat.hair.values, dat.blood.values, dat.cheek.values, 0.25), method='Nelder-Mead')


# In[ ]:

opt.minimize(likelihood_model_A_two_bots, np.array((0.5, 10)), args=(dat.hair.values, dat.blood.values, dat.cheek.values, 0.1), method='Nelder-Mead')


# # ran calculations on server, MCMC over f-fraction and N1

# In[ ]:

pos_unif = pd.read_csv('positions_N_uniform.txt', names=['lnprob', 'f', 'N1'])
frac_burnin = 0.2
pos_unif = pos_unif.iloc[int(pos_unif.shape[0]*frac_burnin+0.5):,]


# In[ ]:

pos_unif


# In[ ]:

pos_unif.hist(grid=False, layout=(1,3), figsize=(16,4))


# # dealing with *all* hairs
# 
# The above dealt only with the mean frequency amongst hairs sampled for a particular individual!

# In[3]:

allh = pd.read_excel('all_hair_freqs.xlsx', 0)
del allh['som']
allh['idx'] = np.concatenate(allh.groupby('individual_id').apply(lambda x: np.arange(x.shape[0])).values)
allh.set_index(['individual_id', 'idx'], inplace=True)


# In[4]:

somf = pd.read_excel('indiv_cheek_hair.xlsx', 0)
somf.head(2)


# In[5]:

allh = allh.reset_index().merge(somf, left_on='individual_id', right_on='individual_id').set_index(['individual_id', 'idx'])


# In[6]:

allh.head()


# The likelihood for individual $i$ is, with hairs $h_i$ is
# 
# $$
# P(h_i \mid s_i) = \sum_{k=0}^{N_1} P(B_1 = k \mid s_i) \prod_{j = 1}^{|h_i|} \sum_{l=0}^{N_2} P(B_2 = l \mid B_1 = k)f(h_{ij} \mid B_2 = l)
# $$

# In[7]:

model_args = zip(*allh.groupby(level=1).apply(lambda x: [x['hair'].values, x['cheek'].iloc[0], x['blood'].iloc[0]]).tolist())


# In[8]:

import likelihood_model as likm
likm = reload(likm)
import scipy.stats as st
import scipy.special


# In[11]:

from __future__ import division
from numba import jit

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


def log_likelihood_model_B_two_bots(params, *args):
    f, N1, N2 = params
    if f < 0 or f > 1:
        return np.inf
    if N1 < 1:
        return np.inf
    if N2 < 1:
        return np.inf
    N1a = int(np.floor(N1))
    N1b = int(np.ceil(N1))
    N2a = int(np.floor(N2))
    N2b = int(np.ceil(N2))
    
    frac_b_N1 = (N1-N1a)/(N1b-N1a)
    frac_b_N2 = (N2-N2a)/(N2b-N2a)
    loglike_aa = log_likelihood_model_B((f, N1a, N2a), *args)
    loglike_ab = log_likelihood_model_B((f, N1a, N2b), *args)
    loglike_ba = log_likelihood_model_B((f, N1b, N2a), *args)
    loglike_bb = log_likelihood_model_B((f, N1b, N2b), *args)
    
    weight_aa = (1-frac_b_N1)*(1-frac_b_N2)
    weight_ab = (1-frac_b_N1)*(frac_b_N2)
    weight_ba = (frac_b_N1)*(1-frac_b_N2)
    weight_bb = (frac_b_N1)*(frac_b_N2)
    weights = [weight_aa, weight_ab, weight_ba, weight_bb]
    loglikes = [loglike_aa, loglike_ab, loglike_ba, loglike_bb]
    val = scipy.special.logsumexp(loglikes, b=weights)
    return val


# In[12]:

log_likelihood_model_B_two_bots((0.5, 20.5, 10.5), *model_args)

