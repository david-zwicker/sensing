'''
Created on Jan 5, 2016

@author: David Zwicker <dzwicker@seas.harvard.edu>
'''

from __future__ import division

import numpy as np
from scipy import integrate, stats, special

from binary_response.sparse_mixtures.lib_spr_theory import LibrarySparseLogNormal
from .pc_base import PrimacyCodingMixin
from utils.math_distributions import lognorm_mean_var



class PrimacyCodingTheory(PrimacyCodingMixin, LibrarySparseLogNormal):
    
    parameters_default = {
        'excitation_distribution': 'gaussian', 
        'excitation_threshold_method': 'integrate'
    }


    def _excitation_distribution(self):
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


    def en_order_statistics_approx(self, Nr, Nc):
        """
        approximates the expected value of the Nc-th variable of the order
        statistics of Nr numbers.
        
        The code for the expectation value is inspired by
            http://stats.stackexchange.com/q/9001/88762
        
        The approximation of the standard deviation has been copied from
            'Accurate approximation to the extreme order statistics of gaussian 
            samples'
            C.-C. Chen and C. W. Tyler
            Commun. Statist. Simula. 28 177-188 (1999)
        """
        # get the distribution of the excitations
        en_dist = self._excitation_distribution()
        
        # approximate the order statistics
        #alpha = np.pi / 8
        #gamma = (Nc - alpha)/(Nr - 2*alpha + 1)
        gamma = Nc / Nr
        en_order_mean = en_dist.ppf(gamma)
        
        # approximate the standard deviation using a formula for the standard
        # deviation of the minimum of Nr Gaussian variables. This overestimates
        # the true standard deviation. 
        en_order_std = 0.5 * (en_dist.ppf(0.8832**(1/Nr))
                              - en_dist.ppf(0.2142**(1/Nr)))
        
        return en_order_mean, en_order_std
    
    
    def en_order_statistics_integrate(self, Nr, Nc, check_norm=True):
        """
        calculates the expected value and the associated standard deviation of
        the Nc-th variable of the order statistics of Nr numbers.
        
        `check_norm` determines whether an additional integral is performed
            to check whether the norm of the probability distribution of the
            order statistics is unity. This is a test for the accuracy of the
            integration routine.
        """
        en_dist = self._excitation_distribution()
        
        def distribution_func(x, n, k, x_power):
            """ definition of the distribution function of the order stats """
            prefactor = k * special.binom(n, k)
            Fx = en_dist.cdf(x) # 0.5 * special.erfc(-x / np.sqrt(2))
            fx = en_dist.pdf(x) # np.exp(-(x**2 / 2))/ np.sqrt(2*np.pi)
            return prefactor * x**x_power * Fx**(k - 1) * (1 - Fx)**(n - k) * fx
        
        # determine the integration interval 
        mean, std = self.en_order_statistics_approx(Nr, Nc)
        int_min = mean - 10*std
        int_max = mean + 10*std
#         xs = stats.norm(mean, std).ppf(np.linspace(0, 1, 501)[1:-1])
#         
#         def distribution_func_inf(x_power):
#             """ function that performs the integration """
#             ys = [distribution_func(x, Nr, Nc, x_power) for x in xs]
#             return integrate.simps(ys, xs)
#         
        def distribution_func_inf(x_power):
            """ function that performs the integration """
            return integrate.quad(distribution_func, int_min, int_max,
                                  args=(Nr, Nc, x_power), limit=1000)[0]
        
        if check_norm:
            # get the norm of the distribution to test the integration routine
            norm = distribution_func_inf(0)
            
            if not np.isclose(norm, 1):
                raise RuntimeError('Integration did not converge for `norm` '
                                   'and resulted in %g for Nr=%d, Nc=%d'
                                   % (norm, Nr, Nc))
        
        # calculate the expected value of the order statistics
        mean = distribution_func_inf(1)

        # calculate the expected value of the order statistics
        M2 = distribution_func_inf(2)
                            
        return mean, np.sqrt(M2 - mean**2)


    def excitation_threshold(self, Nr=None, Nc=None, method='auto'):
        """ returns the approximate excitation threshold that receptors have to
        overcome to be part of the activation pattern.
        
        Depending on the chosen method, this function either returns the
        expected value or a tuple consisting of the expected value and the
        associated standard deviation.
        """
        # setup function arguments
        if Nr is None:
            Nr = self.Nr
        if Nc is None:
            Nc = self.coding_receptors
        if method == 'auto':
            method = self.parameters['excitation_threshold_method']
        
        # determine which method to use for getting the order statistics
        if method == 'integrate':
            en_order_statistics = self.en_order_statistics_integrate
        elif method == 'approx':
            en_order_statistics = self.en_order_statistics_approx
        else:
            raise ValueError('Unknown method `%s` for calculating the '
                             'excitation threshold.' % method)
            
        # calculate the threshold
        return en_order_statistics(Nr, Nr - Nc)


    #===========================================================================
    # OVERWRITE METHODS OF THE BINARY RESPONSE MODEL
    #===========================================================================


    def get_optimal_parameters(self):
        raise NotImplementedError


    def receptor_activity(self):
        """ return the probability with which a single receptor is activated 
        by typical mixtures """
        raise NotImplementedError
        
        
    def receptor_crosstalk(self):
        """ calculates the average activity of the receptor as a response to 
        single ligands. """
        raise NotImplementedError


    def mutual_information(self):
        """ calculates the typical mutual information """
        raise NotImplementedError
    