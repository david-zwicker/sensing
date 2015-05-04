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
from . import library_numeric
from utils.numba_patcher import NumbaPatcher, check_return_value_approx


NUMBA_NOPYTHON = True #< globally decide whether we use the nopython mode

# initialize the numba patcher and add methods one by one
numba_patcher = NumbaPatcher(module=library_numeric)



@numba.jit(nopython=NUMBA_NOPYTHON) 
def LibraryContinuousNumeric_mutual_information_monte_carlo_numba(
        Ns, Nr, steps, int_mat, c_mean, ak, prob_a):
    """ calculate the mutual information using a monte carlo strategy. The
    number of steps is given by the model parameter 'monte_carlo_steps' """
        
    # sample mixtures according to the probabilities of finding
    # substrates
    for _ in xrange(steps):
        # choose a mixture vector according to substrate probabilities
        ak[:] = 0  #< activity pattern of this mixture
        for i in xrange(Ns):
            pi = np.random.exponential() * c_mean[i]
            for a in xrange(Nr):
                ak[a] += int_mat[a, i] * pi
        
        # calculate the activity pattern id
        a_id, base = 0, 1
        for a in xrange(Nr):
            if ak[a] == 1:
                a_id += base
            base *= 2
        
        # increment counter for this output
        prob_a[a_id] += 1
        
    # normalize the probabilities by the number of steps we did
    for k in xrange(len(prob_a)):
        prob_a[k] /= steps
    
    # calculate the mutual information from the observed probabilities
    MI = 0
    for pa in prob_a:
        if pa > 0:
            MI -= pa*np.log2(pa)
    
    return MI
    

def LibraryContinuousNumeric_mutual_information(self, ret_prob_activity=False):
    """ calculate the mutual information by constructing all possible
    mixtures """
    prob_a = np.zeros(2**self.Nr) 
 
    # call the jitted function
    MI = LibraryContinuousNumeric_mutual_information_monte_carlo_numba(
        self.Ns, self.Nr, int(self.parameters['monte_carlo_steps']), 
        self.int_mat,
        self.get_concentration_means(), #< c_mean
        np.empty(self.Nr, np.uint), #< ak
        prob_a
    )
    
    if ret_prob_activity:
        return MI, prob_a
    else:
        return MI


numba_patcher.register_method(
    'LibraryContinuousNumeric.mutual_information',
    LibraryContinuousNumeric_mutual_information,
    check_return_value_approx
)

