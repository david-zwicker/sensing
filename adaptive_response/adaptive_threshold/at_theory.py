'''
Created on Jan 5, 2016

@author: David Zwicker <dzwicker@seas.harvard.edu>
'''

from __future__ import division

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
            
            
    def excitation_threshold(self):
        """ returns the average excitation threshold that receptors have to
        overcome to be part of the activation pattern. """
        en_dist = self.excitation_distribution()
        return self.threshold_factor * en_dist.mean()


    def activity_distance_uncorrelated(self):
        """ calculate the expected difference (Hamming distance) between the
        activity pattern of two completely uncorrelated mixtures.
        """
        p_a = self.excitation_distribution().sf(self.threshold_factor)
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
        return en_dist.sf(self.excitation_threshold())
            

    def receptor_crosstalk(self):
        """ calculates the average activity of the receptor as a response to 
        single ligands. """
        raise NotImplementedError


    def mutual_information(self):
        """ calculates the typical mutual information """
        # TODO: estimate correlations and incorporate this knowledge into I
        raise NotImplementedError
            
            