'''
Created on Dec 29, 2015

@author: David Zwicker <dzwicker@seas.harvard.edu>
'''

from __future__ import division, absolute_import

import numpy as np

from ..binary_response.sparse_mixtures.lib_spr_numeric import \
    LibrarySparseNumeric
from .pc_base import PrimacyCodingMixin
from utils.math import take_popcount



def nlargest_indices(arr, n):
    """
    Return the indices of the `n` largest elements of `arr`.
    This uses a rather slow implementation in python, which compiles well with
    numba. We use this version to have consistency with the numba version.
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



class PrimacyCodingNumeric(PrimacyCodingMixin, LibrarySparseNumeric):
    """ represents a single receptor library that handles sparse mixtures that
    encode their signal using the `coding_receptors` most active receptors """
            

    def excitation_threshold_monte_carlo(self, ret_upper=True):
        """ calculates the average excitation that is necessary to excite
        receptors
        
        `ret_upper` is a flag that determines whether the upper threshold (based
            on the smallest excitation still included in the primacy set) or the
            lower threshold (based on the largest excitation not included in the
            primacy set) is returned
        """
        S_ni = self.sens_mat
        mean = 0
        M2 = 0
                
        for step, c_i in enumerate(self._sample_mixtures(), 1):
            # get the threshold value for this sample
            e_n = np.dot(S_ni, c_i)
            if ret_upper:
                thresh = np.sort(e_n)[-self.coding_receptors]
            else:
                thresh = np.sort(e_n)[-self.coding_receptors - 1]
            
            # accumulate the statistics
            delta = thresh - mean
            mean += delta / step
            M2 += delta * (thresh - mean)
            
        if step < 2:
            std = np.nan
        else:
            std = np.sqrt(M2 / (step - 1))
            
        return mean, std
            
            
    def excitation_threshold_histogram(self, bins, ret_upper=True, steps=None):
        """ calculates the average excitation that is necessary to excite
        receptors
        
        `bins` defines the bins of the histogram (assuming ascending order). The
            returned counts array has one entry more, corresponding to
            thresholds beyond the last entry in `bins`
        `ret_upper` is a flag that determines whether the upper threshold (based
            on the smallest excitation still included in the primacy set) or the
            lower threshold (based on the largest excitation not included in the
            primacy set) is returned
        `steps` determines how many mixtures are sampled
        """
        S_ni = self.sens_mat
        Nc = self.coding_receptors
        counts = np.zeros(len(bins) + 1)
                
        for c_i in self._sample_mixtures(steps=steps):
            # get the threshold value for this sample
            e_n = np.dot(S_ni, c_i)
            if ret_upper:
                thresh = np.sort(e_n)[-Nc]
            else:
                thresh = np.sort(e_n)[-Nc - 1]
            
            idx = np.searchsorted(bins, thresh)
            counts[idx] += 1
            
        return counts
            
            
    def activation_pattern_for_mixture(self, c_i):
        """ returns the receptors that are activated for the mixture `c_i` """
        # calculate excitation
        e_n = np.dot(self.sens_mat, c_i)
        # return the indices of the strongest receptors
        return nlargest_indices(e_n, self.coding_receptors)
            
            
    #===========================================================================
    # OVERWRITE METHODS OF THE BINARY RESPONSE MODEL
    #===========================================================================


    def _sample_activity_indices(self, steps=None):
        """ sample activity vector. Returns indices of active channels """
        S_ni = self.sens_mat

        # iterate over mixtures and yield corresponding activities
        for c_i in self._sample_mixtures(steps):
            e_n = np.dot(S_ni, c_i)
            yield nlargest_indices(e_n, self.coding_receptors)


    def receptor_activity_monte_carlo(self, ret_correlations=False):
        """ calculates the average activity of each receptor """
        # prevent integer overflow in collecting activity patterns
        S_ni = self.sens_mat

        r_n = np.zeros(self.Nr)
        if ret_correlations:
            r_nm = np.zeros((self.Nr, self.Nr))
        
        for c_i in self._sample_mixtures():
            e_n = np.dot(S_ni, c_i)
            a_ni = nlargest_indices(e_n, self.coding_receptors)
            r_n[a_ni] += 1
            if ret_correlations:
                r_nm[a_ni[:, None], a_ni[None, :]] += 1
            
        r_n /= self._sample_steps
        if ret_correlations:
            r_nm /= self._sample_steps
            return r_n, r_nm
        else:
            return r_n


    def receptor_activity_estimate(self, ret_correlations=False,
                                   excitation_model='default', clip=False):
        """ estimates the average activity of each receptor """
        raise NotImplementedError
 
        
    def receptor_crosstalk_estimate(self, ret_receptor_activity=False,
                                    excitation_model='default', clip=False):
        """ calculates the average activity of the receptor as a response to 
        single ligands. """
        raise NotImplementedError


    def mutual_information_monte_carlo(self, ret_prob_activity=False):
        """ calculate the mutual information using a monte carlo strategy. The
        number of steps is given by the model parameter 'monte_carlo_steps' """
        # prevent integer overflow in collecting activity patterns
        assert self.Nr <= self.parameters['max_num_receptors'] <= 63

        base = 2 ** np.arange(0, self.Nr)

        # sample mixtures according to the probabilities of finding
        # substrates
        count_a = np.zeros(2**self.Nr)
        for c in self._sample_mixtures():
            # get the excitation vector ...
            e_n = np.dot(self.sens_mat, c)
            
            # ... determine the activity ...
            a_ni = nlargest_indices(e_n, self.coding_receptors)
            
            # ... and represent it as a single integer
            a_id = base[a_ni].sum()
            
            # increment counter for this output
            count_a[a_id] += 1
            
        # count_a contains the number of times output pattern a was observed.
        # We can thus construct P_a(a) from count_a. 
        q_n = count_a / count_a.sum()
        
        # calculate the mutual information from the result pattern
        MI = -sum(q*np.log2(q) for q in q_n if q != 0)

        if ret_prob_activity:
            return MI, take_popcount(q_n, self.coding_receptors)
        else:
            return MI

                
    def mutual_information_estimate_fast(self):
        """ returns a simple estimate of the mutual information for the special
        case that ret_prob_activity=False, excitation_model='default',
        mutual_information_method='default', and clip=True.
        """
        raise NotImplementedError
                
        