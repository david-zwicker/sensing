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
from utils.math.distributions import lognorm_mean_var, gamma_mean_var

try:
    from adaptive_response.primacy_coding.numba_speedup_numeric import (_activity_distance_tb_lognorm_integrand_numba,
                                _activity_distance_m_lognorm_integrand_numba)
except ImportError:
    _activity_distance_tb_lognorm_integrand_numba = None
    _activity_distance_m_lognorm_integrand_numba = None



class PrimacyCodingTheory(PrimacyCodingMixin, LibrarySparseLogNormal):
    """ class for theoretical calculations where all sensitivities are drawn
    from the same distribution """

    parameters_default = {
        'excitation_distribution': 'log-normal', 
        'excitation_threshold_method': 'approx',
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
        
        if  excitation_dist == 'gaussian' or excitation_dist == 'normal':
            return stats.norm(en_stats['mean'], en_stats['std'])
        elif  excitation_dist == 'log-normal':
            return lognorm_mean_var(en_stats['mean'], en_stats['var'])
        elif  excitation_dist == 'gamma':
            return gamma_mean_var(en_stats['mean'], en_stats['var'])
        else:
            raise ValueError("Unknown excitation distribution `%s`. Supported "
                             "are ['normal', 'log-normal', 'gamma']"
                             % excitation_dist)
            
            
    def _excitation_statistics_single_ligand(self, assume_present=True):
        """ return the excitation statistics when a single ligand comes in
        
        `assume_present` is a flag that determines whether the statistics should
            be calculated assuming that the ligand is always present.
            Alternatively, when the value is set to `False`, we assume that the
            ligand is only present with probability p, given by
            self.substrate_probabilities
        """
        if not self.is_homogeneous_mixture:
            logging.warning('Activity distances can only be estimated for '
                            'homogeneous mixtures, where all ligands have the '
                            'same concentration distribution. We are thus '
                            'using the means of the concentration means and '
                            'variances.')

        # concentration statistics of the single ligand
        if assume_present:
            c_mean = self.c_means.mean()
            c_var = self.c_vars.mean()
            c2_mean = c_mean**2 + c_var
        else:
            p = self.substrate_probabilities.mean()  # prob. of being present
            c_mean_pres = self.c_means.mean()  # mean concentration if present
            c_var_pres = self.c_vars.mean()    # variance if present
            c_mean = p * c_mean_pres
            c_var = p * ((1 - p) * c_mean_pres**2 + c_var_pres)
            c2_mean = p * (c_mean_pres**2 + c_var_pres)

        # statistics of the sensitivity matrix
        S_stats = self.sensitivity_stats()
    
        # calculate statistics of the sum e_n = \sum_i S_ni * c_i        
        en_mean = S_stats['mean'] * c_mean
        en_var  = S_stats['mean']**2 * c_var + S_stats['var'] * c2_mean
        return en_mean, en_var


    def en_order_statistics_approx(self, n, en_dist=None):
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
        arg = (n - alpha) / (self.Nr - 2*alpha + 1)

        # get the distribution of the excitations
        if en_dist is None:
            en_dist = self.excitation_distribution()
        en_order_mean = en_dist.ppf(arg)
        
        # approximate the standard deviation using a formula for the standard
        # deviation of the minimum of Nr Gaussian variables. This overestimates
        # the true standard deviation. 
        en_order_std = 0.5 * (en_dist.ppf(0.8832**(1/self.Nr))
                              - en_dist.ppf(0.2142**(1/self.Nr)))
        
        return en_order_mean, en_order_std
    
    
    def en_order_statistics_dist(self, n, en_dist=None):
        """ returns the distribution of the n-th order statistics using the
        underlying distribution `en_dist` """
        if en_dist is None:
            en_dist = self.excitation_distribution()

        # prefactor of the distribution
        Nr = self.Nr
        pre = n * special.binom(Nr, n)  # = Nr! / (n - 1)! / (Nr - n)!
        
        class order_dist(stats.rv_continuous):
            def _pdf(self,x):
                Fx = en_dist.cdf(x)  # cdf of the excitations
                fx = en_dist.pdf(x)  # pdf of the excitations
                return pre * Fx**(n - 1) * (1 - Fx)**(Nr - n) * fx

        return order_dist(a=en_dist.a, b=en_dist.b, name='order statistics')
    
    
    def en_order_statistics_integrate(self, n, check_norm=True, en_dist=None):
        """
        calculates the expected value and the associated standard deviation of
        the n-th variable of the order statistics of self.Nr excitations
        
        `check_norm` determines whether an additional integral is performed
            to check whether the norm of the probability distribution of the
            order statistics is unity. This is a test for the accuracy of the
            integration routine.
        """
        if en_dist is None:
            en_dist = self.excitation_distribution()
        
        # prefactor of the distribution
        Nr = self.Nr
        pre = n * special.binom(Nr, n)  # = Nr! / (n - 1)! / (Nr - n)!
        
        def distribution_func(x, power=0):
            """
            definition of the distribution function of power-th moment of the
            n-th order statistics of Nr variables
            """
            Fx = en_dist.cdf(x)  # cdf of the excitations
            fx = en_dist.pdf(x)  # pdf of the excitations
            return pre * x**power * Fx**(n - 1) * (1 - Fx)**(Nr - n) * fx
        
        def distribution_func_integrate(power):
            """ function that performs the integration """
            return integrate.quad(distribution_func, en_dist.a, en_dist.b,
                                  args=(power,), limit=1000)[0]
        
        if check_norm:
            # get the norm of the distribution to test the integration routine
            norm = distribution_func_integrate(power=0)
            
            if not np.isclose(norm, 1):
                raise RuntimeError('Integration did not converge for `norm` '
                                   'and resulted in %g for %d-th order '
                                   'statistics ' % (norm, n))
        
        # calculate the expected value of the order statistics
        mean = distribution_func_integrate(power=1)

        # calculate the expected value of the order statistics
        M2 = distribution_func_integrate(power=2)
        var = M2 - mean**2
                            
        return mean, np.sqrt(var)


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
                    - self.coding_receptors * (Nr + 1 - 2*alpha)
                    ) / Nr
        # This complex expression follows from the fact that the expectation of
        # the n-th excitation is
        #     <e_(n)> = F^-1((n - alpha)/(Nr + 1 - 2*alpha))
        # and that the condition Nc = \sum_n P(a_n) = Nr*(1 - F(gamma)) implies
        #     gamma = F^-1(1 - Nc/Nr)
        # To be able to express gamma = <e_(n_thresh)>, we have to equate the
        # arguments of F^-1, which leads to the expression above
        #
        # For alpha = 0, this reduces to the simple form
        #     n_thresh = (1 + 1/self.Nr) * (self.Nr - self.coding_receptors)
        # which is close to the naive expectation
        #     n_thresh = self.Nr - self.coding_receptors
        return n_thresh
    

    def excitation_threshold(self, method='auto', corr_term='approx',
                             en_dist=None):
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
            whether the upper (value 0), lower (value 1), or an intermediated
            excitation threshold is considered.
        `en_dist` can specify a particular excitation distribution. If `None`
            it is estimated based on the ensemble average over odors
        """
        if method == 'auto':
            method = self.parameters['excitation_threshold_method']
        
        # calculate the threshold
        if corr_term == 'approx':
            n_thresh = self._excitation_threshold_order()
        else:
            n_thresh = self.Nr - self.coding_receptors + corr_term
        
        # determine which method to use for getting the order statistics
        if method == 'approx':
            return self.en_order_statistics_approx(n_thresh, en_dist=en_dist)
        elif method == 'integrate':
            return self.en_order_statistics_integrate(n_thresh, en_dist=en_dist)
        else:
            raise ValueError('Unknown method `%s` for calculating the '
                             'excitation threshold.' % method)
            
            
    def activity_distance_uncorrelated(self):
        """ calculate the expected difference (Hamming distance) between the
        activity pattern of two completely uncorrelated mixtures.
        """
        Nc = self.coding_receptors
        return 2 * Nc * (1 - Nc / self.Nr)
    
            
    def _activity_distance_from_distributions_quad(self,
                          en_dist_background, en_dist_target, gamma_1, gamma_2):
        """ numerically solves the integrals for the probabilities of a channel
        becoming active and inactive. Returns the two probabilities.
        
        This calculation has been separate into its own method, so it can be
        easily overwritten with a numba-accelerated one """   
        # determine the probability that a channel turns on
        def integrand_on(e_1):
            return (en_dist_target.sf(gamma_2 - e_1) *
                    en_dist_background.pdf(e_1))
        p_on = integrate.quad(integrand_on, 0, gamma_1)[0]

        # determine the probability that a channel turns off
        if gamma_2 > gamma_1:
            def integrand_off(e_1):
                return (en_dist_target.cdf(gamma_2 - e_1) *
                        en_dist_background.pdf(e_1))
            p_off = integrate.quad(integrand_off, gamma_1, gamma_2)[0]
        else:
            p_off = 0
            
        return p_on, p_off
                

    def _activity_distance_from_distributions(self, en_dist_background,
                                              en_dist_target, en_dist_sum=None):
        """ calculates the expected difference between the activities associated
        with a background odor and the mixture of that odor with a target. The
        excitation probability distributions are given as parameters and it is
        assumed that all excitations are uncorrelated
        
        We thus compare the following two odors:
            odor 1: background
            odor 2: sum = background + target 
        
        `en_dist_sum` is the excitation distribution of the sum. If it is not
            given it is approximated by a log-normal distribution with mean and
            variance given by the sum of the respective values of the two other
            distributions
        """
        if en_dist_sum is None:
            # determine distribution of the sum if it is not given
            en_sum_mean = en_dist_background.mean() + en_dist_target.mean()
            en_sum_var =  en_dist_background.var() + en_dist_target.var()
            en_dist_sum = lognorm_mean_var(en_sum_mean, en_sum_var)
            
        # determine the excitation thresholds
        gamma_1, _ = self.excitation_threshold(en_dist=en_dist_background)
        gamma_2, _ = self.excitation_threshold(en_dist=en_dist_sum)
        if gamma_2 < gamma_1:
            # gamma_2 >= gamma_1 is assumed below
            logging.warning('Threshold with target is smaller than without '
                            '(%g < %g)', gamma_2, gamma_1)
        
        # call the integration routine
        p_on, p_off = self._activity_distance_from_distributions_quad(
                        en_dist_background, en_dist_target, gamma_1, gamma_2)
        
        return self.Nr * (p_on + p_off)
        
            
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
            return self.activity_distance_uncorrelated
        
        # get the excitation distributions of different mixture sizes
        en_mean, en_var = self._excitation_statistics_single_ligand()
        en_dist_b = lognorm_mean_var(en_mean, en_var)
        en_dist_t = lognorm_mean_var(c_ratio * en_mean, c_ratio**2 * en_var)
        
        # determine the expected activity distance
        return self._activity_distance_from_distributions(en_dist_b, en_dist_t)        
                
            
    def activity_distance_mixture_size(self, mixture_size):
        """ calculate the expected difference (Hamming distance) between the
        activity pattern of a mixture of size `mixture_size` and the same
        mixture plus an additional ligand. The concentrations of the ligands are
        chosen according to the current concentration statistics.
        """
        # handle some special cases to avoid numerical problems at the
        # integration boundaries
        if mixture_size < 0:
            raise ValueError('Mixture size must be positive.')
        elif mixture_size == 0:
            return self.activity_distance_uncorrelated()
        elif np.isinf(mixture_size):
            return 0
        
        # get the excitation distributions of different mixture sizes
        en_mean, en_var = self._excitation_statistics_single_ligand()
        en_dist_b = lognorm_mean_var(mixture_size * en_mean,
                                     mixture_size * en_var)
        en_dist_t = lognorm_mean_var(en_mean, en_var)
        
        # determine the expected activity distance
        return self._activity_distance_from_distributions(en_dist_b, en_dist_t)        
                

    def activity_distance_mixture_similarity(self, mixture_size, overlap=0):
        """ calculates the expected Hamming distance between the activation
        pattern of two mixtures with `mixture_size` ligands of equal 
        concentration. `overlap` denotes the number of
        ligands that are the same in the two mixtures """
        if not 0 <= overlap <= mixture_size:
            raise ValueError('Mixture overlap `overlap` must be between 0 and '
                             '`mixture_size`.')
        elif overlap == mixture_size:
            return 0
        elif overlap == 0:
            return self.activity_distance_uncorrelated()
    
        s = mixture_size
        sB = overlap
        sD = (s - sB)  # number of different ligands
    
        # get the excitation distributions of different mixture sizes
        en_mean, en_var = self._excitation_statistics_single_ligand()
        en_dist_total = lognorm_mean_var(s*en_mean, s*en_var)
        en_dist_same = lognorm_mean_var(sB*en_mean, sB*en_var)
        en_dist_diff = lognorm_mean_var(sD*en_mean, sD*en_var)
    
        # determine the excitation thresholds
        gamma, _ = self.excitation_threshold(en_dist=en_dist_total)
    
        # use the general definition of the integral
        def p_change_xor(e_same):
            """ probability that the different ligands of either mixture
            bring the excitation above threshold """ 
            # probability that the excitation does not exceed threshold
            cdf_val = en_dist_diff.cdf(gamma - e_same)
            return cdf_val * (1 - cdf_val) * en_dist_same.pdf(e_same)
    
        # TODO: replace this using numba?
        # look at `_activity_distance_m_lognorm_integrand_numba`
        p_different = integrate.quad(p_change_xor, en_dist_same.a, gamma)[0]
    
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
        raise NotImplementedError
            
            