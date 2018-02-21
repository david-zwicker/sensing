'''
Created on Jan 5, 2016

@author: David Zwicker <dzwicker@seas.harvard.edu>
'''

from __future__ import division

import logging

import numpy as np
from scipy import integrate, stats, special

from binary_response.sparse_mixtures.lib_spr_theory import LibrarySparseLogNormal

from .pc_base import PrimacyCodingMixin
from utils.math.distributions import lognorm_mean_var

try:
    from .numba_speedup import (_activity_distance_tb_lognorm_integrand_numba,
                                _activity_distance_m_lognorm_integrand_numba)
except ImportError:
    _activity_distance_tb_lognorm_integrand_numba = None
    _activity_distance_m_lognorm_integrand_numba = None



class PrimacyCodingTheory(PrimacyCodingMixin, LibrarySparseLogNormal):
    """ class for theoretical calculations where all sensitivities are drawn
    from the same distribution """

    parameters_default = {
        'excitation_distribution': 'log-normal', 
        'excitation_threshold_method': 'integrate',
        'order_statistics_alpha': np.pi / 8,
        # two common choices for `order_statistics_alpha` are
        #     0: H. A. David and H. N. Nagaraja, Order statistics, Wiley (1970)
        #            see section 4.5, in particular equation (4.5.1)
        #     np.pi/8: see http://stats.stackexchange.com/q/9001/88762
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
            

    def en_order_statistics_approx(self, n):
        """
        approximates the expected value of the n-th variable of the order
        statistics of the Nr excitations. Here, n runs from 1 .. Nr.
        
        The code for the expectation value is inspired by
            http://stats.stackexchange.com/q/9001/88762
        
        The approximation of the standard deviation has been copied from
            'Accurate approximation to the extreme order statistics of gaussian 
            samples'
            C.-C. Chen and C. W. Tyler
            Commun. Statist. Simula. 28 177-188 (1999)
        """
        # approximate the order statistics
        alpha = self.parameters['order_statistics_alpha']
        gamma = (n - alpha) / (self.Nr - 2*alpha + 1)

        # get the distribution of the excitations
        en_dist = self.excitation_distribution()
        en_order_mean = en_dist.ppf(gamma)
        
        # approximate the standard deviation using a formula for the standard
        # deviation of the minimum of Nr Gaussian variables. This overestimates
        # the true standard deviation. 
        en_order_std = 0.5 * (en_dist.ppf(0.8832**(1/self.Nr))
                              - en_dist.ppf(0.2142**(1/self.Nr)))
        
        return en_order_mean, en_order_std
    
    
    def en_order_statistics_integrate(self, n, check_norm=True):
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
            prefactor = k * special.binom(n, k)  # = n!/(k - 1)!/(n - k)!
            
            Fx = en_dist.cdf(x)  # cdf of the excitations
            fx = en_dist.pdf(x)  # pdf of the excitations
            return prefactor * x**x_power * Fx**(k - 1) * (1 - Fx)**(n - k) * fx
            # TODO: Check whether this integrand is correct
        
        # determine the integration interval from the approximation
        mean, std = self.en_order_statistics_approx(n)
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


    def _excitation_threshold_order(self):
        """ return the (real-valued) index in the order statistics of the
        excitations that corresponds to the excitation threshold. The index is
        calculated such that \sum_n a_n = Nc on average if approximate order
        statistics are used. This does not (yet) work for integrated order
        statistics
        """
        # if the approximate order statistics are used 
        Nr = self.Nr
        alpha = self.parameters['order_statistics_alpha']
        n_thresh = (Nr * (1 + Nr - alpha)
                    - self.coding_receptors * (1 + Nr - 2*alpha)
                    ) / Nr
        # For alpha = 0, this reduces to the simple form
        #     n_thresh = (1 + 1/self.Nr) * (self.Nr - self.coding_receptors)
        # which is close to naive expectation
        #     n_thresh = self.Nr - self.coding_receptors
        return n_thresh


    def excitation_threshold(self, method='auto', corr_term='approx'):
        """ returns the approximate excitation threshold that receptors have to
        overcome to be part of the activation pattern.
        
        Returns a tuple consisting of the expected value and the associated
        standard deviation.
        
        `method` can be either `integrate` in which case the expectation value
            of the threshold is calculated by integrating the distribution of
            the order statistics. Alternative, when method == `approx`, this
            integral is approximated. The default value `auto` chooses the
            method based on the parameter `excitation_threshold_method`.
        `corr_term` is the correcting term between 0 and 1, that determines
            whether the lower or upper excitation threshold is considered.
        """
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
        if corr_term == 'approx':
            return en_order_statistics(self._excitation_threshold_order()) 
        else:
            return en_order_statistics(self.Nr - self.coding_receptors
                                       + corr_term)
            
            
    def activity_distance_uncorrelated(self):
        """ calculate the expected difference (Hamming distance) between the
        activity pattern of two completely uncorrelated mixtures.
        """
        Nc = self.coding_receptors
        return 2*Nc*(1 - Nc/self.Nr)
        
            
    def activity_distance_target_background_approx_small(self, c_ratio):
        """ calculate the expected difference (Hamming distance) between the
        activity pattern of a single ligand and this ligand plus a second one
        at a concentration `c_ratio` times the concentration of the first one.
        
        This function approximated the distance for small `c_ratios`
        """
        en_dist = self.excitation_distribution()
        en_thresh = self.excitation_threshold('approx')[0]
        
        # estimate the probability that a receptor was inactive and gets
        # activated
        p_on = (c_ratio
                * en_dist.pdf(en_thresh)
                * integrate.quad(en_dist.sf, en_thresh, en_dist.b)[0])
        
        # estimate the probability that a receptor was active and gets shut down
        Nc = self.coding_receptors
        Nr = self.Nr
        p_off = c_ratio * Nc/Nr * (1 - Nc/Nr)
        
        return Nr*(p_on + p_off)
            
            
    def activity_distance_target_background(self, c_ratio):
        """ calculate the expected difference (Hamming distance) between the
        activity pattern of a single ligand and this ligand plus a second one
        at a concentration `c_ratio` times the concentration of the first one.
        """
        # handle some special cases to avoid numerical problems at the
        # integration boundaries
        if c_ratio < 0:
            raise ValueError('Concentration ratio `c_ratio` must be positive.')
        elif c_ratio == 0:
            return 0
        elif np.isinf(c_ratio):
            return self.mixture_distance_uncorrelated()
        
        # determine the excitation thresholds
        p_inact = 1 - self.coding_receptors / self.Nr
        en_dist = self.excitation_distribution()
        e_thresh_0 = en_dist.ppf(p_inact)
        e_thresh_rho = (1 + c_ratio) * e_thresh_0
        
        # determine the probability of changing the activity of a receptor
        if (self.parameters['excitation_distribution'] == 'log-normal'
            and _activity_distance_tb_lognorm_integrand_numba):
            # use the numba enhanced integrand
            integrand = _activity_distance_tb_lognorm_integrand_numba
            args = (c_ratio, e_thresh_rho, en_dist.mean(), en_dist.var())
            
        else:
            # use the general definition of the integral
            def integrand(e1):
                """ integrand for the activation probability """ 
                cdf_val = en_dist.cdf((e_thresh_rho - e1) / c_ratio)
                return cdf_val * en_dist.pdf(e1)
            
            args = ()
            
        p_on = p_inact - integrate.quad(integrand, en_dist.a, e_thresh_0,
                                        args=args)[0]
        p_off = integrate.quad(integrand, e_thresh_0, en_dist.b,
                               args=args)[0]
    
        return self.Nr * (p_on + p_off)
                

    def activity_distance_mixtures(self, mixture_size, mixture_overlap=0):
        """ calculates the expected Hamming distance between the activation
        pattern of two mixtures with `mixture_size` ligands of equal 
        concentration. `mixture_overlap` denotes the number of
        ligands that are the same in the two mixtures """
        if not 0 <= mixture_overlap <= mixture_size:
            raise ValueError('Mixture overlap `mixture_overlap` must be '
                             'between 0 and `mixture_size`.')
        elif mixture_overlap == mixture_size:
            return 0
        elif mixture_overlap == 0:
            return self.activity_distance_uncorrelated()
    
        s = mixture_size
        sB = mixture_overlap
        sD = (s - sB) # number of different ligands
        
        if not self.is_homogeneous_mixture:
            logging.warning('Activity distances can only be estimated for '
                            'homogeneous mixtures, where all ligands have the '
                            'same concentration distribution. We are thus '
                            'using the means of the concentration means and '
                            'variances.')

        c_mean = self.c_means.mean()
        c_var = self.c_vars.mean()
        S_stats = self.sensitivity_stats()
        en_mean = S_stats['mean'] * c_mean
        en_var = (S_stats['mean']**2 + S_stats['var']) * c_var
    
        # get the excitation distributions of different mixture sizes
        en_dist_total = lognorm_mean_var(s*en_mean, s*en_var)
    
        # determine the excitation thresholds
        p_inact = 1 - self.coding_receptors / self.Nr
        e_thresh_total = en_dist_total.ppf(p_inact)

        # determine the probability of changing the activity of a receptor
        if (self.parameters['excitation_distribution'] == 'log-normal'
            and _activity_distance_m_lognorm_integrand_numba):
            # use the numba enhanced integrand
            p_change_xor = _activity_distance_m_lognorm_integrand_numba
            args = (e_thresh_total, sB, sD, en_mean, en_var)
            
        else:
            # use the general definition of the integral
            en_dist_same = lognorm_mean_var(sB*en_mean, sB*en_var)
            en_dist_diff = lognorm_mean_var(sD*en_mean, sD*en_var)

            # use the general definition of the integral
            def p_change_xor(e_same):
                """ probability that the different ligands of either mixture
                bring the excitation above threshold """ 
                # prob that the excitation does not exceed threshold
                cdf_val = en_dist_diff.cdf(e_thresh_total - e_same) 
                return cdf_val * (1 - cdf_val) * en_dist_same.pdf(e_same)

            args = ()
    
        # integrate over all excitations of the common ligands
        p_different = integrate.quad(p_change_xor, en_dist_total.a,
                                     e_thresh_total, args=args)[0]
    
        return 2 * self.Nr * p_different
    
    
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
            
            