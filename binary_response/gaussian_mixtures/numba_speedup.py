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
from . import lib_gau_numeric
from utils.numba_patcher import NumbaPatcher, check_return_value_approx


NUMBA_NOPYTHON = True #< globally decide whether we use the nopython mode
NUMBA_NOGIL = True

# initialize the numba patcher and add methods one by one
numba_patcher = NumbaPatcher(module=lib_gau_numeric)



@numba.jit(nopython=NUMBA_NOPYTHON, nogil=NUMBA_NOGIL) 
def LibraryGaussianNumeric_receptor_activity_numba(Ns, Nr, steps, int_mat,
                                                     c_means, alpha, count_a):
    """ calculate the mutual information using a monte carlo strategy. The
    number of steps is given by the model parameter 'monte_carlo_steps' """
        
    # sample mixtures according to the probabilities of finding
    # substrates
    for _ in range(steps):
        # choose a mixture vector according to substrate probabilities
        alpha[:] = 0  #< activity pattern of this mixture
        for i in range(Ns):
            ci = np.random.exponential() * c_means[i]
            for a in range(Nr):
                alpha[a] += int_mat[a, i] * ci
        
        # calculate the activity pattern id
        for a in range(Nr):
            if alpha[a] >= 1:
                count_a[a] += 1
    

def LibraryGaussianNumeric_receptor_activity(self):
    """ calculate the mutual information by constructing all possible
    mixtures """
    count_a = np.zeros(self.Nr, np.uint32) 
    steps = self._sample_steps
 
    # call the jitted function
    LibraryGaussianNumeric_receptor_activity_numba(
        self.Ns, self.Nr, self._sample_steps, self.int_mat,
        self.concentration_means, #< c_means
        np.empty(self.Nr, np.double), #< alpha
        count_a
    )
    
    return count_a / steps


numba_patcher.register_method(
    'LibraryGaussianNumeric.receptor_activity',
    LibraryGaussianNumeric_receptor_activity,
    check_return_value_approx
)



@numba.jit(nopython=NUMBA_NOPYTHON, nogil=NUMBA_NOGIL) 
def LibraryGaussianNumeric_mutual_information_numba(Ns, Nr, steps, int_mat,
                                                      c_means, alpha, prob_a):
    """ calculate the mutual information using a monte carlo strategy. The
    number of steps is given by the model parameter 'monte_carlo_steps' """
        
    # sample mixtures according to the probabilities of finding
    # substrates
    for _ in range(steps):
        # choose a mixture vector according to substrate probabilities
        alpha[:] = 0  #< activity pattern of this mixture
        for i in range(Ns):
            ci = np.random.exponential() * c_means[i]
            for a in range(Nr):
                alpha[a] += int_mat[a, i] * ci
        
        # calculate the activity pattern id
        a_id, base = 0, 1
        for a in range(Nr):
            if alpha[a] >= 1:
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
    

def LibraryGaussianNumeric_mutual_information(self, ret_prob_activity=False):
    """ calculate the mutual information by constructing all possible
    mixtures """
    prob_a = np.zeros(2**self.Nr) 
 
    # call the jitted function
    MI = LibraryGaussianNumeric_mutual_information_numba(
        self.Ns, self.Nr, self._sample_steps, 
        self.int_mat,
        self.concentration_means, #< c_means
        np.empty(self.Nr, np.double), #< alpha
        prob_a
    )
    
    if ret_prob_activity:
        return MI, prob_a
    else:
        return MI


numba_patcher.register_method(
    'LibraryGaussianNumeric.mutual_information',
    LibraryGaussianNumeric_mutual_information,
    check_return_value_approx
)

