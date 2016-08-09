'''
Created on Jan 16, 2015

@author: David Zwicker <dzwicker@seas.harvard.edu>

Module that monkey patches classes in other modules with equivalent, but faster
methods.
'''

from __future__ import division

import functools
import logging

import numba
import numpy as np

# these methods are used in getattr calls
from . import pc_numeric
from binary_response.sparse_mixtures.numba_speedup import \
                        LibrarySparseNumeric_excitation_statistics_monte_carlo
from utils.math import take_popcount
from utils.math.distributions import lognorm_mean_var_to_mu_sigma
from utils.numba.patcher import (NumbaPatcher, check_return_value_approx,
                                 check_return_dict_approx)
from utils.numba.tools import nlargest_indices_numba, lognorm_cdf, lognorm_pdf


NUMBA_NOPYTHON = True #< globally decide whether we use the nopython mode
NUMBA_NOGIL = True

# initialize the numba patcher and add methods one by one
numba_patcher = NumbaPatcher(module=pc_numeric)



@numba.jit(nopython=True)
def _activity_distance_tb_lognorm_integrand_numba(e1, c_ratio, e_thresh_rho,
                                                  en_mean, en_var):
    """ evaluates the integrand for calculating the mixture distance with
    log-normally distributed excitations """
    if e1 == 0:
        return 0
    cdf_arg = (e_thresh_rho - e1) / c_ratio
    if cdf_arg <= 0:
        return 0

    # evaluate cdf of the log-normal distribution
    cdf_val = lognorm_cdf(cdf_arg, en_mean, en_var)

    # evaluate pdf of the log-normal distribution
    pdf_val = lognorm_pdf(e1, en_mean, en_var)
    
    return cdf_val * pdf_val



@numba.jit(nopython=True)
def _activity_distance_m_lognorm_integrand_numba(e_same, e_thresh_total, sB, sD,
                                                 en_mean, en_var):
    """ probability that the different ligands of either mixture
    bring the excitation above threshold """ 
    # prob that the excitation does not exceed threshold
    cdf_val = lognorm_cdf(e_thresh_total - e_same, sD*en_mean, sD*en_var) 

    return cdf_val * (1 - cdf_val) * lognorm_pdf(e_same, sB*en_mean, sB*en_var)



# copy the accelerated method from the binary_response package
numba_patcher.register_method(
    'PrimacyCodingNumeric.excitation_statistics_monte_carlo',
    LibrarySparseNumeric_excitation_statistics_monte_carlo,
    functools.partial(check_return_dict_approx, rtol=0.1, atol=0.1)
)



excitation_threshold_monte_carlo_numba_template = """ 
def function(steps, Nr_k, S_ni, p_i, c_means, c_spread):
    ''' calculate the mutual information using a monte carlo strategy. The
    number of steps is given by the model parameter 'monte_carlo_steps' '''
    Nr, Ns = S_ni.shape
    e_n = np.empty(Nr, np.double)
    mean, M2 = 0, 0

    # sample mixtures according to the probabilities of finding ligands
    for step in range(1, steps + 1):
        # choose a mixture vector according to substrate probabilities
        e_n[:] = 0  #< activity pattern of this mixture
        for i in range(Ns):
            if np.random.random() < p_i[i]:
                # mixture contains substrate i
                {CONCENTRATION_GENERATOR}
                for n in range(Nr):
                    e_n[n] += S_ni[n, i] * c_i
        
        # calculate the excitation threshold
        indices = nlargest_indices_numba(e_n, Nr_k)
        thresh = e_n[indices].min()

        # accumulate the statistics
        delta = thresh - mean
        mean += delta / step
        M2 += delta * (thresh - mean)

    if steps < 2:
        std = np.nan
    else:
        std = np.sqrt(M2 / (step - 1))
        
    return mean, std
"""


def PrimacyCodingNumeric_excitation_threshold_monte_carlo_numba_generator(conc_gen):
    """ generates a function that calculates the receptor activity for a given
    concentration generator """
    func_code = excitation_threshold_monte_carlo_numba_template.format(
        CONCENTRATION_GENERATOR=conc_gen)
    # make sure all necessary objects are in the scope
    scope = {'np': np, 'nlargest_indices_numba': nlargest_indices_numba} 
    exec(func_code, scope)
    func = scope['function']
    return numba.jit(nopython=NUMBA_NOPYTHON, nogil=NUMBA_NOGIL)(func)


PrimacyCodingNumeric_excitation_threshold_monte_carlo_expon_numba = \
    PrimacyCodingNumeric_excitation_threshold_monte_carlo_numba_generator(
        "c_i = np.random.exponential() * c_means[i]")
    
# Note that the parameter c_mean is actually the mean of the underlying normal
# distribution
PrimacyCodingNumeric_excitation_threshold_monte_carlo_lognorm_numba = \
    PrimacyCodingNumeric_excitation_threshold_monte_carlo_numba_generator(
        "c_i = np.random.lognormal(c_means[i], c_spread[i])")
    
PrimacyCodingNumeric_excitation_threshold_monte_carlo_bernoulli_numba = \
    PrimacyCodingNumeric_excitation_threshold_monte_carlo_numba_generator(
        "c_i = c_means[i]")
    
    

def PrimacyCodingNumeric_excitation_threshold_monte_carlo(
                                               self, ret_correlations=False):
    """ calculate the mutual information by constructing all possible
    mixtures """
    fixed_mixture_size = self.parameters['fixed_mixture_size']
    if self.is_correlated_mixture or fixed_mixture_size is not None:
        logging.warning('Numba code not implemented for correlated mixtures. '
                        'Falling back to pure-python method.')
        this = PrimacyCodingNumeric_excitation_threshold_monte_carlo
        return this._python_function(self, ret_correlations)
 
    # call the jitted function
    c_distribution = self.parameters['c_distribution']
    if c_distribution == 'exponential':
        r = PrimacyCodingNumeric_excitation_threshold_monte_carlo_expon_numba(
            self.monte_carlo_steps,  
            self.coding_receptors, self.sens_mat,
            self.substrate_probabilities, #< p_i
            self.c_means, 0               #< concentration statistics
        )
    
    elif c_distribution == 'log-normal':
        mus, sigmas = lognorm_mean_var_to_mu_sigma(self.c_means, self.c_vars,
                                                   'numpy')
        r = PrimacyCodingNumeric_excitation_threshold_monte_carlo_lognorm_numba(
            self.monte_carlo_steps,  
            self.coding_receptors, self.sens_mat,
            self.substrate_probabilities, #< p_i
            mus, sigmas                   #< concentration statistics
        )

    elif c_distribution == 'bernoulli':
        r = PrimacyCodingNumeric_excitation_threshold_monte_carlo_bernoulli_numba(
            self.monte_carlo_steps,  
            self.coding_receptors, self.sens_mat,
            self.substrate_probabilities, #< p_i
            self.c_means, 0               #< concentration statistics
        )
        
    else:
        logging.warning('Numba code is not implemented for distribution `%s`. '
                        'Falling back to pure-python method.', c_distribution)
        this = PrimacyCodingNumeric_excitation_threshold_monte_carlo
        return this._python_function(self, ret_correlations)
    
    return r
    
    
numba_patcher.register_method(
    'PrimacyCodingNumeric.excitation_threshold_monte_carlo',
    PrimacyCodingNumeric_excitation_threshold_monte_carlo,
    functools.partial(check_return_value_approx, rtol=0.1, atol=0.1)
)

    
    
receptor_activity_monte_carlo_numba_template = """ 
def function(steps, Nr_k, S_ni, p_i, c_means, c_spread, ret_correlations, r_n,
             r_nm):
    ''' calculate the mutual information using a monte carlo strategy. The
    number of steps is given by the model parameter 'monte_carlo_steps' '''
    Nr, Ns = S_ni.shape
    e_n = np.empty(Nr, np.double)

    # sample mixtures according to the probabilities of finding ligands
    for _ in range(steps):
        # choose a mixture vector according to substrate probabilities
        e_n[:] = 0  #< activity pattern of this mixture
        for i in range(Ns):
            if np.random.random() < p_i[i]:
                # mixture contains substrate i
                {CONCENTRATION_GENERATOR}
                for n in range(Nr):
                    e_n[n] += S_ni[n, i] * c_i
        
        # calculate the activity pattern
        a_ids = nlargest_indices_numba(e_n, Nr_k)
        for n in a_ids:
            r_n[n] += 1
        
        if ret_correlations:
            # calculate the correlations
            for n in a_ids:
                for m in a_ids:
                    r_nm[n, m] += 1
"""


def PrimacyCodingNumeric_receptor_activity_monte_carlo_numba_generator(conc_gen):
    """ generates a function that calculates the receptor activity for a given
    concentration generator """
    func_code = receptor_activity_monte_carlo_numba_template.format(
        CONCENTRATION_GENERATOR=conc_gen)
    # make sure all necessary objects are in the scope
    scope = {'np': np, 'nlargest_indices_numba': nlargest_indices_numba} 
    exec(func_code, scope)
    func = scope['function']
    return numba.jit(nopython=NUMBA_NOPYTHON, nogil=NUMBA_NOGIL)(func)


PrimacyCodingNumeric_receptor_activity_monte_carlo_expon_numba = \
    PrimacyCodingNumeric_receptor_activity_monte_carlo_numba_generator(
        "c_i = np.random.exponential() * c_means[i]")
    
# Note that the parameter c_mean is actually the mean of the underlying normal
# distribution
PrimacyCodingNumeric_receptor_activity_monte_carlo_lognorm_numba = \
    PrimacyCodingNumeric_receptor_activity_monte_carlo_numba_generator(
        "c_i = np.random.lognormal(c_means[i], c_spread[i])")
    
PrimacyCodingNumeric_receptor_activity_monte_carlo_bernoulli_numba = \
    PrimacyCodingNumeric_receptor_activity_monte_carlo_numba_generator(
        "c_i = c_means[i]")
    
    

def PrimacyCodingNumeric_receptor_activity_monte_carlo(
                                               self, ret_correlations=False):
    """ calculate the mutual information by constructing all possible
    mixtures """
    fixed_mixture_size = self.parameters['fixed_mixture_size']
    if self.is_correlated_mixture or fixed_mixture_size is not None:
        logging.warning('Numba code not implemented for correlated mixtures. '
                        'Falling back to pure-python method.')
        this = PrimacyCodingNumeric_receptor_activity_monte_carlo
        return this._python_function(self, ret_correlations)

    r_n = np.zeros(self.Nr) 
    r_nm = np.zeros((self.Nr, self.Nr)) 
    steps = self.monte_carlo_steps
 
    # call the jitted function
    c_distribution = self.parameters['c_distribution']
    if c_distribution == 'exponential':
        PrimacyCodingNumeric_receptor_activity_monte_carlo_expon_numba(
            steps,  
            self.coding_receptors, self.sens_mat,
            self.substrate_probabilities, #< p_i
            self.c_means, 0,              #< concentration statistics
            ret_correlations,
            r_n, r_nm
        )
    
    elif c_distribution == 'log-normal':
        mus, sigmas = lognorm_mean_var_to_mu_sigma(self.c_means, self.c_vars,
                                                   'numpy')
        PrimacyCodingNumeric_receptor_activity_monte_carlo_lognorm_numba(
            steps,  
            self.coding_receptors, self.sens_mat,
            self.substrate_probabilities, #< p_i
            mus, sigmas,                  #< concentration statistics
            ret_correlations,
            r_n, r_nm
        )

    elif c_distribution == 'bernoulli':
        PrimacyCodingNumeric_receptor_activity_monte_carlo_bernoulli_numba(
            steps,  
            self.coding_receptors, self.sens_mat,
            self.substrate_probabilities, #< p_i
            self.c_means, 0,              #< concentration statistics
            ret_correlations,
            r_n, r_nm
        )
        
    else:
        logging.warning('Numba code is not implemented for distribution `%s`. '
                        'Falling back to pure-python method.', c_distribution)
        this = PrimacyCodingNumeric_receptor_activity_monte_carlo
        return this._python_function(self, ret_correlations)
    
    # return the normalized output
    r_n /= steps
    if ret_correlations:
        r_nm /= steps
        return r_n, r_nm
    else:
        return r_n


numba_patcher.register_method(
    'PrimacyCodingNumeric.receptor_activity_monte_carlo',
    PrimacyCodingNumeric_receptor_activity_monte_carlo,
    check_return_value_approx
)



mutual_information_monte_carlo_numba_template = ''' 
def function(steps, Nr_k, S_ni, p_i, c_means, c_spread, prob_a):
    """ calculate the mutual information using a monte carlo strategy. The
    number of steps is given by the model parameter 'monte_carlo_steps' """
    Nr, Ns = S_ni.shape
    e_n = np.empty(Nr, np.double)
    
    # sample mixtures according to the probabilities of finding
    # substrates
    for _ in range(steps):
        # choose a mixture vector according to substrate probabilities
        e_n[:] = 0  #< reset excitation pattern of this mixture
        for i in range(Ns):
            if np.random.random() < p_i[i]:
                # mixture contains substrate i => choose c_i
                {CONCENTRATION_GENERATOR}
                for n in range(Nr):
                    e_n[n] += S_ni[n, i] * c_i
        
        # calculate the activity pattern id
        a_id = 0
        for k in nlargest_indices_numba(e_n, Nr_k):
            a_id += (1 << k) #< same as 2**k
        
        # increment counter for this output
        prob_a[a_id] += 1
        
    # normalize the probabilities by the number of steps we did
    prob_a /= steps
    
    # calculate the mutual information from the observed probabilities
    MI = 0
    for pa in prob_a:
        if pa > 0:
            MI -= pa*np.log2(pa)
    
    return MI
'''


def PrimacyCodingNumeric_mutual_information_monte_carlo_numba_generator(conc_gen):
    """ generates a function that calculates the receptor activity for a given
    concentration generator """
    func_code = mutual_information_monte_carlo_numba_template.format(
        CONCENTRATION_GENERATOR=conc_gen)
    # make sure all necessary objects are in the scope
    scope = {'np': np, 'nlargest_indices_numba': nlargest_indices_numba} 
    exec(func_code, scope)
    func = scope['function']
    return numba.jit(nopython=NUMBA_NOPYTHON, nogil=NUMBA_NOGIL)(func)


PrimacyCodingNumeric_mutual_information_monte_carlo_expon_numba = \
    PrimacyCodingNumeric_mutual_information_monte_carlo_numba_generator(
        "c_i = np.random.exponential() * c_means[i]")
    
# Note that the parameter c_mean is actually the mean of the underlying normal
# distribution
PrimacyCodingNumeric_mutual_information_monte_carlo_lognorm_numba = \
    PrimacyCodingNumeric_mutual_information_monte_carlo_numba_generator(
        "c_i = np.random.lognormal(c_means[i], c_spread[i])")
    
PrimacyCodingNumeric_mutual_information_monte_carlo_bernoulli_numba = \
    PrimacyCodingNumeric_mutual_information_monte_carlo_numba_generator(
        "c_i = c_means[i]")
    


def PrimacyCodingNumeric_mutual_information_monte_carlo(
                                                self, ret_prob_activity=False):
    """ calculate the mutual information by constructing all possible
    mixtures """
    if self.is_correlated_mixture:
        logging.warning('Numba code not implemented for correlated mixtures. '
                        'Falling back to pure-python method.')
        this = PrimacyCodingNumeric_mutual_information_monte_carlo
        return this._python_function(self, ret_prob_activity)

    # prevent integer overflow in collecting activity patterns
    assert self.Nr <= self.parameters['max_num_receptors'] <= 63

    prob_a = np.zeros(2**self.Nr)
    
    # call the jitted function
    c_distribution = self.parameters['c_distribution']
    if c_distribution == 'exponential':
        MI = PrimacyCodingNumeric_mutual_information_monte_carlo_expon_numba(                                
            self.monte_carlo_steps,  
            self.coding_receptors, self.sens_mat,
            self.substrate_probabilities, #< p_i
            self.c_means, 0,              #< concentration statistics
            prob_a
        )
        
    elif c_distribution == 'log-normal':
        mus, sigmas = lognorm_mean_var_to_mu_sigma(self.c_means, self.c_vars,
                                                   'numpy')
        MI = PrimacyCodingNumeric_mutual_information_monte_carlo_lognorm_numba(
            self.monte_carlo_steps,  
            self.coding_receptors, self.sens_mat,
            self.substrate_probabilities, #< p_i
            mus, sigmas,                  #< concentration statistics
            prob_a
        )        
        
    elif c_distribution == 'bernoulli':
        MI = PrimacyCodingNumeric_mutual_information_monte_carlo_bernoulli_numba(
            self.monte_carlo_steps,  
            self.coding_receptors, self.sens_mat,
            self.substrate_probabilities, #< p_i
            self.c_means, 0,              #< concentration statistics
            prob_a
        )        
        
    else:
        logging.warning('Numba code is not implemented for distribution `%s`. '
                        'Falling back to pure-python method.', c_distribution)
        this = PrimacyCodingNumeric_mutual_information_monte_carlo
        return this._python_function(self, ret_prob_activity)
    
    if ret_prob_activity:
        return MI, take_popcount(prob_a, self.coding_receptors)
    else:
        return MI


numba_patcher.register_method(
    'PrimacyCodingNumeric.mutual_information_monte_carlo',
    PrimacyCodingNumeric_mutual_information_monte_carlo,
    check_return_value_approx
)
