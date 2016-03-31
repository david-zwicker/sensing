'''
Created on Feb 22, 2016

@author: David Zwicker <dzwicker@seas.harvard.edu>
'''

from __future__ import division, absolute_import

import numpy as np

from binary_response.sparse_mixtures.lib_spr_numeric import LibrarySparseNumeric
from .at_base import AdaptiveThresholdMixin
from utils.misc import StatisticsAccumulator



class AdaptiveThresholdNumeric(AdaptiveThresholdMixin, LibrarySparseNumeric):
    """ represents a single receptor library that handles sparse mixtures that
    where receptors get active if their excitation is above a fraction of the
    total excitation """
    
    
    def excitation_statistics_monte_carlo_pure(self, ret_correlations=False):
        """ 
        calculates the statistics of the excitation of the receptors.
        Returns the mean excitation, the variance, and the covariance matrix.
        This function just calculates the statistics of unnormalized
        excitations, which is implemented in the parent function
        We implemented this as a separate function so it can selectively be
        replaced with a version that is sped up by numba         
        """
        parent = super(AdaptiveThresholdNumeric, self)
        return parent.excitation_statistics_monte_carlo(ret_correlations)
    
    
    def excitation_statistics_monte_carlo(self, ret_correlations=False,
                                          normalized=False):
        """
        calculates the statistics of the excitation of the receptors.
        Returns the mean excitation, the variance, and the covariance matrix.
        
        The algorithms used here have been taken from
            https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance
        """
        if not normalized:
            return self.excitation_statistics_monte_carlo_pure(ret_correlations)
            
        S_ni = self.sens_mat
        S_ni_mean = S_ni.mean()

        # initialize the statistics calculation
        stats = StatisticsAccumulator(ret_cov=ret_correlations)

        # sample mixtures and safe the requested data
        for c_i in self._sample_mixtures():
            e_n = np.dot(S_ni, c_i)
            e_n /= c_i.sum() * S_ni_mean #< normalize
            stats.add(e_n)

        # return the requested statistics
        if ret_correlations:
            try:
                enm_cov = stats.cov
            except RuntimeError:
                enm_cov = np.full((self.Nr, self.Nr), np.nan, np.double)
            en_var = np.diag(enm_cov)
            return {'mean': stats.mean, 'std': np.sqrt(en_var), 'var': en_var,
                    'cov': enm_cov}
        else:        
            en_var = stats.var 
            return {'mean': stats.mean, 'std': np.sqrt(en_var), 'var': en_var}
                

    def excitation_threshold_statistics(self):
        """ returns the statistics of the excitation threshold that receptors
        have to overcome to be part of the activation pattern.
        
        `normalized` determines whether the statistics of the excitations are
            calculated for normalized concentration vector 
        """
        S_ni = self.sens_mat
        alpha = self.threshold_factor

        e_thresh_stats = StatisticsAccumulator()

        # iterate over samples and collect information about the threshold        
        for c_i in self._sample_mixtures():
            e_n = np.dot(S_ni, c_i)
            e_thresh = alpha * e_n.mean()
            e_thresh_stats.add(e_thresh)

        return {'mean': e_thresh_stats.mean,
                'var': e_thresh_stats.var,
                'std': e_thresh_stats.std}
                    
                    
    def _sample_activities(self, steps=None):
        """ sample activity vectors """
        S_ni = self.sens_mat
        alpha = self.threshold_factor

        # iterate over mixtures and yield corresponding activities
        for c_i in self._sample_mixtures(steps):
            e_n = np.dot(S_ni, c_i)
            a_n = (e_n >= alpha * e_n.mean())
            yield a_n
            
    
    def receptor_activity_monte_carlo(self, ret_correlations=False):
        """ calculates the average activity of each receptor """
        # prevent integer overflow in collecting activity patterns
        assert self.Nr <= self.parameters['max_num_receptors'] <= 63

        r_n = np.zeros(self.Nr)
        if ret_correlations:
            r_nm = np.zeros((self.Nr, self.Nr))
        
        for a_n in self._sample_activities():
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

        # sample mixtures according to the probabilities of finding
        # substrates
        count_a = np.zeros(2**self.Nr)
        for a_n in self._sample_activities():
            # represent activity as a single integer
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
        
    
    def mutual_information_estimate_fast(self):
        """ not implemented for adaptive thresholds """ 
        raise NotImplementedError
        