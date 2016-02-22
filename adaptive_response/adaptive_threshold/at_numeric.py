'''
Created on Feb 22, 2016

@author: David Zwicker <dzwicker@seas.harvard.edu>
'''

from __future__ import division, absolute_import

import numpy as np

from binary_response.sparse_mixtures.lib_spr_numeric import LibrarySparseNumeric
from .at_base import AdaptiveThresholdMixin



class AdaptiveThresholdNumeric(AdaptiveThresholdMixin, LibrarySparseNumeric):
    """ represents a single receptor library that handles sparse mixtures that
    where receptors get active if their excitation is above a fraction of the
    total excitation """
            
    
    def receptor_activity_monte_carlo(self, ret_correlations=False):
        """ calculates the average activity of each receptor """
        # prevent integer overflow in collecting activity patterns
        assert self.Nr <= self.parameters['max_num_receptors'] <= 63

        S_ni = self.sens_mat
        alpha = self.threshold_factor

        r_n = np.zeros(self.Nr)
        if ret_correlations:
            r_nm = np.zeros((self.Nr, self.Nr))
        
        for c_i in self._sample_mixtures():
            e_n = np.dot(S_ni, c_i)
            a_n = (e_n >= alpha * e_n.mean())
            r_n[a_n] += 1
            if ret_correlations:
                r_nm[np.outer(a_n, a_n)] += 1
            
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


    def activation_pattern_for_mixture(self, c_i):
        """ returns the receptors that are activated for the mixture `c_i` """
        # calculate excitation
        e_n = np.dot(self.sens_mat, c_i)
        a_n = (e_n >= self.threshold_factor * e_n.mean())
        # return the indices of the active receptors
        return np.flatnonzero(a_n)
            
            
    def mutual_information_monte_carlo(self, ret_prob_activity=False):
        """ calculate the mutual information using a monte carlo strategy. The
        number of steps is given by the model parameter 'monte_carlo_steps' """
        # prevent integer overflow in collecting activity patterns
        assert self.Nr <= self.parameters['max_num_receptors'] <= 63

        base = 2 ** np.arange(0, self.Nr)
        S_ni = self.sens_mat
        alpha = self.threshold_factor

        # sample mixtures according to the probabilities of finding
        # substrates
        count_a = np.zeros(2**self.Nr)
        for c_i in self._sample_mixtures():
            
            # get the activity vector ...
            e_n = np.dot(S_ni, c_i)
            a_n = (e_n >= alpha * e_n.mean())
            # ... and represent it as a single integer
            a_id = np.dot(base, a_n)
            # increment counter for this output
            count_a[a_id] += 1
            
        # count_a contains the number of times output pattern a was observed.
        # We can thus construct P_a(a) from count_a. 
        q_n = count_a / count_a.sum()
        
        # calculate the mutual information from the result pattern
        MI = -sum(q*np.log2(q) for q in q_n if q != 0)

        if ret_prob_activity:
            return MI, q_n
        else:
            return MI