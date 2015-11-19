'''
Created on Sep 18, 2015

@author: David Zwicker <dzwicker@seas.harvard.edu>
'''

from __future__ import division

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



def random_seed(seed=None):
    """ sets the seed of the random number generator of numpy and numba """
    np.random.seed(seed)
    if numba and seed is not None:
        numba_random_seed(seed)
    