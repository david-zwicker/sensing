'''
Created on Jan 16, 2015

@author: David Zwicker <dzwicker@seas.harvard.edu>

Module that monkey patches classes in other modules with equivalent, but faster
methods.
'''

from __future__ import division

import logging
import math
import numba
import numpy as np

# these methods are used in getattr calls
from . import lib_spr_numeric
from utils.numba_patcher import (NumbaPatcher, check_return_value_approx,
                                 check_return_value_exact)


NUMBA_NOPYTHON = True #< globally decide whether we use the nopython mode
NUMBA_NOGIL = True

# initialize the numba patcher and add methods one by one
numba_patcher = NumbaPatcher(module=lib_spr_numeric)



@numba.jit(nopython=NUMBA_NOPYTHON, nogil=NUMBA_NOGIL) 
def LibrarySparseNumeric_receptor_activity_monte_carlo_numba(
        Ns, Nr, steps, S_ni, p_i, d_i, a_n, ret_correlations, r_n, r_nm):
    """ calculate the mutual information using a monte carlo strategy. The
    number of steps is given by the model parameter 'monte_carlo_steps' """
        
    # sample mixtures according to the probabilities of finding ligands
    for _ in range(steps):
        # choose a mixture vector according to substrate probabilities
        a_n[:] = 0  #< activity pattern of this mixture
        for i in range(Ns):
            if np.random.random() < p_i[i]:
                # mixture contains substrate i
                c_i = np.random.exponential() * d_i[i]
                for n in range(Nr):
                    a_n[n] += S_ni[n, i] * c_i
        
        # calculate the activity pattern 
        for n in range(Nr):
            if a_n[n] >= 1:
                r_n[n] += 1
                
        if ret_correlations:
            # calculate the correlations
            for n in range(Nr):
                if a_n[n] >= 1:
                    r_nm[n, n] += 1
                    for m in range(n):
                        if a_n[m] >= 1:
                            r_nm[n, m] += 1
                            r_nm[m, n] += 1
                
    

def LibrarySparseNumeric_receptor_activity_monte_carlo(
                                               self, ret_correlations=False):
    """ calculate the mutual information by constructing all possible
    mixtures """
    if self.is_correlated_mixture:
        logging.warn('Not implemented for correlated mixtures. Falling back to '
                     'pure-python method.')
        this = LibrarySparseNumeric_receptor_activity_monte_carlo
        return this._python_function(self, ret_correlations)

    # prevent integer overflow in collecting activity patterns
    assert self.Nr <= self.parameters['max_num_receptors'] <= 63

    r_n = np.zeros(self.Nr) 
    r_nm = np.zeros((self.Nr, self.Nr)) 
    steps = self.monte_carlo_steps
 
    # call the jitted function
    LibrarySparseNumeric_receptor_activity_monte_carlo_numba(
        self.Ns, self.Nr, steps, self.sens_mat,
        self.substrate_probabilities, #< p_i
        self.concentrations,          #< d_i
        np.empty(self.Nr, np.double), #< a_n
        ret_correlations,
        r_n, r_nm
    )
    
    # return the normalized output
    r_n /= steps
    if ret_correlations:
        r_nm /= steps
        return r_n, r_nm
    else:
        return r_n


numba_patcher.register_method(
    'LibrarySparseNumeric.receptor_activity_monte_carlo',
    LibrarySparseNumeric_receptor_activity_monte_carlo,
    check_return_value_approx
)



@numba.jit(nopython=NUMBA_NOPYTHON, nogil=NUMBA_NOGIL) 
def LibrarySparseNumeric_mutual_information_monte_carlo_numba(
                                  Ns, Nr, steps, S_ni, p_i, d_i, a_n, prob_a):
    """ calculate the mutual information using a monte carlo strategy. The
    number of steps is given by the model parameter 'monte_carlo_steps' """
        
    # sample mixtures according to the probabilities of finding
    # substrates
    for _ in range(steps):
        # choose a mixture vector according to substrate probabilities
        a_n[:] = 0  #< activity pattern of this mixture
        for i in range(Ns):
            if np.random.random() < p_i[i]:
                # mixture contains substrate i
                ci = np.random.exponential() * d_i[i]
                for n in range(Nr):
                    a_n[n] += S_ni[n, i] * ci
        
        # calculate the activity pattern id
        a_id, base = 0, 1
        for n in range(Nr):
            if a_n[n] >= 1:
                a_id += base
            base *= 2
        
        # increment counter for this output
        prob_a[a_id] += 1
        
    # normalize the probabilities by the number of steps we did
    for k in range(len(prob_a)):
        prob_a[k] /= steps
    
    # calculate the mutual information from the observed probabilities
    MI = 0
    for pa in prob_a:
        if pa > 0:
            MI -= pa*np.log2(pa)
    
    return MI
    

def LibrarySparseNumeric_mutual_information_monte_carlo(
                                                self, ret_prob_activity=False):
    """ calculate the mutual information by constructing all possible
    mixtures """
    if self.is_correlated_mixture:
        logging.warn('Not implemented for correlated mixtures. Falling back to '
                     'pure-python method.')
        this = LibrarySparseNumeric_mutual_information_monte_carlo
        return this._python_function(self, ret_prob_activity)

    # prevent integer overflow in collecting activity patterns
    assert self.Nr <= self.parameters['max_num_receptors'] <= 63

    prob_a = np.zeros(2**self.Nr)
 
    # call the jitted function
    MI = LibrarySparseNumeric_mutual_information_monte_carlo_numba(
        self.Ns, self.Nr, self.monte_carlo_steps,  self.sens_mat,
        self.substrate_probabilities, #< p_i
        self.concentrations,          #< d_i
        np.empty(self.Nr, np.double), #< a_n
        prob_a
    )
    
    if ret_prob_activity:
        return MI, prob_a
    else:
        return MI


numba_patcher.register_method(
    'LibrarySparseNumeric.mutual_information_monte_carlo',
    LibrarySparseNumeric_mutual_information_monte_carlo,
    check_return_value_approx
)



#@numba.jit(nopython=NUMBA_NOPYTHON, nogil=NUMBA_NOGIL) 
def LibrarySparseNumeric_mutual_information_estimate_fast_numba(
                                                          Ns, Nr, pi, di, S_ni):
    """ returns a simple estimate of the mutual information for the special
    case that ret_prob_activity=False, excitation_model='default',
    mutual_information_method='default', and clip=True.
    """
    
    en_mean = np.zeros(Nr, np.double)
    enm_cov = np.zeros((Nr, Nr), np.double)
    
    # calculate the statistics of the excitation
    for i in range(Ns):
        ci_mean = di[i] * pi[i]
        ci_var = di[i] * ci_mean * (2 - pi[i])
        
        for n in range(Nr):
            en_mean[n] += S_ni[n, i] * ci_mean
            for m in range(n + 1):
                enm_cov[n, m] += S_ni[n, i] * S_ni[m, i] * ci_var

    # calculate the receptor activity
    qn = en_mean #< reuse the memory
    for n in range(Nr):
        if en_mean[n] > 0:
            # mean is zero => qn = 0 (which we do not need to set because it is
            # already zero)
            if enm_cov[n, n] == 0:
                # variance is zero => q_n = Theta(e_n - 1)
                if en_mean[n] >= 1:
                    qn[n] = 1
                else:
                    qn[n] = 0
            else:
                # proper evaluation
                en_cv2 = enm_cov[n, n] / en_mean[n]**2
                enum = math.log(math.sqrt(1 + en_cv2) / en_mean[n])
                denom = math.sqrt(2*math.log(1 + en_cv2))
                qn[n] = 0.5 * math.erfc(enum/denom)
    
    # calculate the crosstalk and the mutual information in one iteration
    prefactor = 8/math.log(2)/(2*np.pi)**2
    
    MI = 0
    for n in range(Nr):
        if 0 < qn[n] < 1:
            MI -= qn[n]*np.log2(qn[n]) + (1 - qn[n])*np.log2(1 - qn[n])
        if enm_cov[n, n] > 0: 
            for m in range(n):
                if enm_cov[m, m] > 0:
                    rho2 = enm_cov[n, m]**2 / (enm_cov[n, n] * enm_cov[m, m])
                    MI -= prefactor * rho2

    # clip the result to [0, Nr]
    if MI < 0:
        return 0
    elif MI > Nr:
        return Nr
    else:
        return MI



def LibrarySparseNumeric_mutual_information_estimate_fast(self):
    """ returns a simple estimate of the mutual information for the special
    case that ret_prob_activity=False, excitation_model='default',
    mutual_information_method='default', and clip=True.
    """
    return LibrarySparseNumeric_mutual_information_estimate_fast_numba(
        self.Ns, self.Nr, self.substrate_probabilities, self.concentrations,
        self.sens_mat
    )

  

numba_patcher.register_method(
    'LibrarySparseNumeric.mutual_information_estimate_fast',
    LibrarySparseNumeric_mutual_information_estimate_fast,
    check_return_value_exact
)
