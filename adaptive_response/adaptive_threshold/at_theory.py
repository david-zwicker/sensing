'''
Created on Jan 5, 2016

@author: David Zwicker <dzwicker@seas.harvard.edu>
'''

from __future__ import division

import numpy as np
from scipy import stats

from binary_response.sparse_mixtures.lib_spr_theory import LibrarySparseLogNormal

from .at_base import AdaptiveThresholdMixin
from utils.math_distributions import lognorm_mean_var



class AdaptiveThresholdTheory(AdaptiveThresholdMixin, LibrarySparseLogNormal):
    """ class for theoretical calculations where all sensitivities are drawn
    from the same distribution """

    parameters_default = {
        'excitation_distribution': 'log-normal', 
    }
            
    
    @property
    def threshold_factor_compensated(self):
        """ returns the threshold factor corrected for the excitation that was
        actually measured """
        alpha = self.threshold_factor
        return alpha * (self.Nr - 1) / (self.Nr - alpha)


    def excitation_distribution(self):
        """ returns a scipy.stats distribution for the excitations with the
        given mean and standard deviation
        """ 
        excitation_dist = self.parameters['excitation_distribution']
        en_stats = self.excitation_statistics()
        
        if  excitation_dist == 'gaussian':
            return stats.norm(en_stats['mean'], en_stats['std'])
        elif  excitation_dist == 'log-normal':
            return lognorm_mean_var(en_stats['mean'], en_stats['var'])
        else:
            raise ValueError("Unknown excitation distribution `%s`. Supported "
                             "are ['gaussian', 'log-normal']" % excitation_dist)
            
            
    def excitation_threshold(self, compensated=False):
        """ returns the average excitation threshold that receptors have to
        overcome to be part of the activation pattern.
        `compensated` determines whether the compensated threshold factor is
            used to determine the excitation threshold
        """
        if compensated:
            alpha = self.threshold_factor_compensated
        else:
            alpha = self.threshold_factor
        
        return  alpha * self.excitation_statistics()['mean']

        
    def excitation_threshold_statistics(self):
        """ returns the statistics of the excitation threshold that receptors
        have to overcome to be part of the activation pattern. """
        alpha = self.threshold_factor
        
        # get statistics of the total concentration c_tot = \sum_i c_i
        ctot_stats = self.ctot_statistics()
        ctot_mean = ctot_stats['mean']
        ctot_var = ctot_stats['var']
        
        # get statistics of the sensitivities S_ni
        S_stats = self.sensitivity_stats()
        S_mean = S_stats['mean']
        S_var = S_stats['var']
        
        # calculate statistics of the mean excitation
        en_mean_mean = ctot_mean * S_mean
        en_mean_var = (S_mean**2 * ctot_var
                       + (ctot_var + ctot_mean / self.Ns) * S_var / self.Nr)
        
        # return the statistics of the excitation threshold
        en_thresh_var = alpha**2 * en_mean_var
        return {'mean': alpha * en_mean_mean,
                'var': en_thresh_var,
                'std': np.sqrt(en_thresh_var)}


    def activity_distance_uncorrelated(self):
        """ calculate the expected difference (Hamming distance) between the
        activity pattern of two completely uncorrelated mixtures.
        """
        alpha_hat = self.threshold_factor_compensated
        p_a = self.excitation_distribution().sf(alpha_hat)
        return 2 * self.Nr * p_a * (1 - p_a)
            
            
    #===========================================================================
    # OVERWRITE METHODS OF THE BINARY RESPONSE MODEL
    #===========================================================================


    def get_optimal_parameters(self):
        """
        returns a guess for the optimal parameters for the sensitivity
        distribution
        """ 
        raise NotImplementedError


    def receptor_activity(self):
        """ return the probability with which a single receptor is activated 
        by typical mixtures """
        en_dist = self.excitation_distribution()
        return en_dist.sf(self.excitation_threshold(compensated=True))
            

    def receptor_crosstalk(self):
        """ calculates the average activity of the receptor as a response to 
        single ligands. """
        raise NotImplementedError


    def mutual_information(self):
        """ calculates the typical mutual information """
        # TODO: estimate correlations and incorporate this knowledge into I
        raise NotImplementedError
            
            