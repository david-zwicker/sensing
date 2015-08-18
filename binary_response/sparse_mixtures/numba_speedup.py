'''
Created on Jan 16, 2015

@author: David Zwicker <dzwicker@seas.harvard.edu>

Module that monkey patches classes in other modules with equivalent, but faster
methods.
'''

from __future__ import division

import numba
import numpy as np

# these methods are used in getattr calls
from . import library_numeric  # @UnresolvedImport
from utils.numba_patcher import NumbaPatcher, check_return_value_approx


NUMBA_NOPYTHON = True #< globally decide whether we use the nopython mode
NUMBA_NOGIL = True

# initialize the numba patcher and add methods one by one
numba_patcher = NumbaPatcher(module=library_numeric)



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
            for n in range(Nr):
                if a_n[n] >= 1:
                    r_nm[n, n] += 1
                    for m in range(n):
                        if a_n[m] >= 1:
                            r_nm[n, m] += 1
                            r_nm[m, n] += 1
                
    

def LibrarySparseNumeric_receptor_activity_monte_carlo(self, ret_correlations=False):
    """ calculate the mutual information by constructing all possible
    mixtures """
    if self.correlated_mixture:
        raise NotImplementedError('Not implemented for correlated mixtures')

    # prevent integer overflow in collecting activity patterns
    assert self.Nr <= self.parameters['max_num_receptors'] <= 63

    r_n = np.zeros(self.Nr) 
    r_nm = np.zeros((self.Nr, self.Nr)) 
    steps = self.monte_carlo_steps
 
    # call the jitted function
    LibrarySparseNumeric_receptor_activity_monte_carlo_numba(
        self.Ns, self.Nr, steps, self.int_mat,
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
def LibrarySparseNumeric_mutual_information_numba(Ns, Nr, steps, S_ni, p_i, d_i,
                                                  a_n, prob_a):
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
    

def LibrarySparseNumeric_mutual_information(self, ret_prob_activity=False):
    """ calculate the mutual information by constructing all possible
    mixtures """
    if self.correlated_mixture:
        raise NotImplementedError('Not implemented for correlated mixtures')

    # prevent integer overflow in collecting activity patterns
    assert self.Nr <= self.parameters['max_num_receptors'] <= 63

    prob_a = np.zeros(2**self.Nr)
 
    # call the jitted function
    MI = LibrarySparseNumeric_mutual_information_numba(
        self.Ns, self.Nr, self.monte_carlo_steps,  self.int_mat,
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
    'LibrarySparseNumeric.mutual_information',
    LibrarySparseNumeric_mutual_information,
    check_return_value_approx
)

