'''
Created on Sep 18, 2015

@author: David Zwicker <dzwicker@seas.harvard.edu>
'''

from __future__ import division

import math

import numpy as np

try:
    import numba
except ImportError:
    numba = None



if numba:
    # define functions for the case where numba is available
    
    @numba.jit(nopython=True, nogil=True)
    def numba_random_seed(seed):
        """ sets the seed of the random number generator of numba """
        np.random.seed(seed)
        
    
    
    @numba.jit(nopython=True, nogil=True)
    def nlargest_indices_numba(arr, n):
        """
        Return the indices of the `n` largest elements of `arr`
        """
        indices = np.arange(n)
        values = np.empty(n)
        values[:] = arr[:n]
        minpos = values.argmin()
        minval = values[minpos]
        
        for k in range(n, len(arr)):
            val = arr[k]
            if val > minval:
                indices[minpos] = k
                values[minpos] = val
                minpos = values.argmin()
                minval = values[minpos]
                
        return indices
    
    
    
    @numba.jit(nopython=True)
    def lognorm_pdf(x, mean, var):
        """ probability distribution function of a log-normal distribution
        parameterized by its `mean` and variance `var` """
        mean2 = mean**2
        exp_enum = np.log(x * np.sqrt(mean2 + var) / mean2) ** 2
        exp_denom = 2. * np.log(1. + var / mean2)
    
        enum = np.exp(-exp_enum / exp_denom)
        denom = x * np.sqrt(2. * np.pi * np.log(1. + (var / mean2)))
        return enum / denom
    
    
    
    @numba.jit(nopython=True)
    def lognorm_cdf(x, mean, var):
        """ cumulative distribution function of a log-normal distribution
        parameterized by its `mean` and variance `var` """
        mean2 = mean**2
        return 0.5 * math.erfc(
            np.log(mean2 / (x*np.sqrt(mean2 + var)))
            /(np.sqrt(2*np.log(1 + var/mean2)))
        )



def random_seed(seed=None):
    """ sets the seed of the random number generator of numpy and numba """
    np.random.seed(seed)
    if numba and seed is not None:
        numba_random_seed(seed)
    