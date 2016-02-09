'''
Created on Jan 5, 2016

@author: David Zwicker <dzwicker@seas.harvard.edu>
'''

from __future__ import division

from functools import partial

import numpy as np
from scipy import integrate, stats, special

from binary_response.sparse_mixtures.lib_spr_theory import LibrarySparseLogNormal

from .pc_base import PrimacyCodingMixin
from utils.math_distributions import lognorm_mean_var



class PrimacyCodingTheory(PrimacyCodingMixin, LibrarySparseLogNormal):
    """ class for theoretical calculations where all sensitivities are drawn
    from the same distribution """

    parameters_default = {
        'excitation_distribution': 'gaussian', 
        'excitation_threshold_method': 'integrate'
    }


    def excitation_distribution(self):
        """ returns a scipy.stats distribution for the excitations with the
        given mean and standard deviation """ 
        excitation_dist = self.parameters['excitation_distribution']
        en_stats = self.excitation_statistics()
        
        if  excitation_dist == 'gaussian':
            return stats.norm(en_stats['mean'], en_stats['std'])
        elif  excitation_dist == 'log-normal':
            return lognorm_mean_var(en_stats['mean'], en_stats['var'])
        else:
            raise ValueError("Unknown excitation distribution `%s`. Supported "
                             "are ['gaussian', 'log-normal']" % excitation_dist)
            

    def en_order_statistics_approx(self, n, alpha=None):
        """
        approximates the expected value of the n-th variable of the order
        statistics of the Nr excitations.
        
        If `alpha` is given, the compensated formula is used with the specified
            alpha. If alpha='auto', alpha = np.pi / 8 is used.
        
        The code for the expectation value is inspired by
            http://stats.stackexchange.com/q/9001/88762
        
        The approximation of the standard deviation has been copied from
            'Accurate approximation to the extreme order statistics of gaussian 
            samples'
            C.-C. Chen and C. W. Tyler
            Commun. Statist. Simula. 28 177-188 (1999)
        """
        if alpha == 'auto':
            alpha = np.pi / 8
        
        # approximate the order statistics
        if alpha:
            gamma = (n - alpha)/(self.Nr - 2*alpha + 1)
        else:
            gamma = n / self.Nr

        # get the distribution of the excitations
        en_dist = self.excitation_distribution()
        en_order_mean = en_dist.ppf(gamma)
        
        # approximate the standard deviation using a formula for the standard
        # deviation of the minimum of Nr Gaussian variables. This overestimates
        # the true standard deviation. 
        en_order_std = 0.5 * (en_dist.ppf(0.8832**(1/self.Nr))
                              - en_dist.ppf(0.2142**(1/self.Nr)))
        
        return en_order_mean, en_order_std
    
    
    def en_order_statistics_integrate(self, n, check_norm=True,
                                      order_stats_alpha=None):
        """
        calculates the expected value and the associated standard deviation of
        the n-th variable of the order statistics of self.Nr excitations
        
        `check_norm` determines whether an additional integral is performed
            to check whether the norm of the probability distribution of the
            order statistics is unity. This is a test for the accuracy of the
            integration routine.
        """
        en_dist = self.excitation_distribution()
        
        def distribution_func(x, n, k, x_power):
            """
            definition of the distribution function of x_power-th moment of the
            k-th order statistics of n variables
            """
            prefactor = k * special.binom(n, k)
            Fx = en_dist.cdf(x) # cdf of the excitations
            fx = en_dist.pdf(x) # pdf of the excitations
            return prefactor * x**x_power * Fx**(k - 1) * (1 - Fx)**(n - k) * fx
        
        # determine the integration interval 
        mean, std = self.en_order_statistics_approx(n, order_stats_alpha)
        int_min = mean - 10*std
        int_max = mean + 10*std

        def distribution_func_inf(x_power):
            """ function that performs the integration """
            return integrate.quad(distribution_func, int_min, int_max,
                                  args=(self.Nr, n, x_power), limit=1000)[0]
        
        if check_norm:
            # get the norm of the distribution to test the integration routine
            norm = distribution_func_inf(0)
            
            if not np.isclose(norm, 1):
                raise RuntimeError('Integration did not converge for `norm` '
                                   'and resulted in %g for n=%d'
                                   % (norm, n))
        
        # calculate the expected value of the order statistics
        mean = distribution_func_inf(1)

        # calculate the expected value of the order statistics
        M2 = distribution_func_inf(2)
                            
        return mean, np.sqrt(M2 - mean**2)


    def excitation_threshold(self, method='auto', order_stats_alpha=None):
        """ returns the approximate excitation threshold that receptors have to
        overcome to be part of the activation pattern.
        
        Depending on the chosen method, this function either returns the
        expected value or a tuple consisting of the expected value and the
        associated standard deviation.
        """
        if method == 'auto':
            method = self.parameters['excitation_threshold_method']
        
        # determine which method to use for getting the order statistics
        if method == 'integrate':
            en_order_statistics = self.en_order_statistics_integrate
        elif method == 'approx':
            en_order_statistics = partial(self.en_order_statistics_approx,
                                          alpha=order_stats_alpha)
        else:
            raise ValueError('Unknown method `%s` for calculating the '
                             'excitation threshold.' % method)
            
        # calculate the threshold
        return en_order_statistics(self.Nr - self.coding_receptors)
    

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
        return self.coding_receptors / self.Nr
            

    def receptor_crosstalk(self):
        """ calculates the average activity of the receptor as a response to 
        single ligands. """
        raise NotImplementedError


    def mutual_information(self):
        """ calculates the typical mutual information """
        # TODO: estimate correlations and incorporate this knowledge into I
        raise NotImplementedError
            
            