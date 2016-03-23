'''
Created on Jan 5, 2016

@author: David Zwicker <dzwicker@seas.harvard.edu>
'''

from __future__ import division

import logging

import numpy as np
from scipy import stats, integrate

from binary_response.sparse_mixtures.lib_spr_base import LibrarySparseBase
from binary_response.sparse_mixtures.lib_spr_theory import LibrarySparseLogNormal

from .at_base import AdaptiveThresholdMixin
from utils.math_distributions import lognorm_mean_var
from utils.misc import xlog2x



class AdaptiveThresholdTheory(AdaptiveThresholdMixin, LibrarySparseLogNormal):
    """ class for theoretical calculations where all sensitivities are drawn
    independently from a log-normal distribution """

    parameters_default = {
        'excitation_distribution': 'log-normal', 
        'compensated_threshold': False,
    }

    
    def concentration_statistics(self, normalized=False):
        """ returns statistics for each individual substrate.
        If `normalized` is True, the statistics will be estimated for the
            normalized concentrations c_i / c_tot, where c_tot = \sum_i c_i
        """
        # get real concentration statistics
        parent = super(AdaptiveThresholdTheory, self)
        ci_stats = parent.concentration_statistics()
        
        if normalized:
            # get statistics of the total concentration
            ctot_mean = ci_stats['mean'].sum()
            ctot_var = ci_stats['var'].sum()
            ctot_eff = ctot_var / ctot_mean**2
            
            # scale the concentration statistics
            chi = (1 + ctot_eff)
            ci_stats['mean'] = ci_stats['mean'] / ctot_mean * chi
            ci_stats['var'] = (chi / ctot_mean)**2 * \
                            (ci_stats['var'] * chi + ci_stats['mean']*ctot_eff)
                                 
            ci_stats['cov'] = np.diag(ci_stats['var'])
            ci_stats['std'] = np.sqrt(ci_stats['var'])            
            
        return ci_stats


    def excitation_statistics(self, normalized=False):
        """ calculates the statistics of the excitation of the receptors.
        Returns the expected mean excitation, the variance, and the covariance
        matrix of any given receptor """
        if self.is_correlated_mixture:
            raise NotImplementedError('Not implemented for correlated mixtures')

        # get statistics of the individual concentrations
        c_stats = self.concentration_statistics(normalized=normalized)
        c2_mean = c_stats['mean']**2 + c_stats['var']
        c2_mean_sum = c2_mean.sum()

        # get statistics of the total concentration c_tot = \sum_i c_i
        if normalized:
            ctot_mean = 1
            ctot_var = 0
        else:
            ctot_mean = c_stats['mean'].sum()
            ctot_var = c_stats['var'].sum()
        
        # get statistics of the sensitivities S_ni
        S_stats = self.sensitivity_stats()
        S_mean = S_stats['mean']
        
        # calculate statistics of the sum e_n = \sum_i S_ni * c_i        
        en_mean = S_mean * ctot_mean
        en_var  = S_mean**2 * ctot_var + S_stats['var'] * c2_mean_sum
        enm_cov = S_mean**2 * ctot_var + S_stats['cov'] * c2_mean_sum

        return {'mean': en_mean, 'std': np.sqrt(en_var), 'var': en_var,
                'cov': enm_cov}
        

    def excitation_distribution(self, normalized=False):
        """ returns a scipy.stats distribution for the excitations with the
        given mean and standard deviation
        """ 
        excitation_dist = self.parameters['excitation_distribution']
        en_stats = self.excitation_statistics(normalized=normalized)
        
        if  excitation_dist == 'gaussian':
            if normalized:
                raise ValueError('Gaussian distributions are not supported for '
                                 'normalized excitations.')
            return stats.norm(en_stats['mean'], en_stats['std'])
        elif  excitation_dist == 'log-normal':
            return lognorm_mean_var(en_stats['mean'], en_stats['var'])
        else:
            raise ValueError("Unknown excitation distribution `%s`. Supported "
                             "are ['gaussian', 'log-normal']" % excitation_dist)
    
    
    def excitation_distribution_mixture(self, concentration=1, mixture_size=1):
        """ returns a scipy.stats distribution for the excitations with the
        given mean and standard deviation
        """ 
        excitation_dist = self.parameters['excitation_distribution']
        S_stats = self.sensitivity_stats()
        en_mean = S_stats['mean'] * mixture_size * concentration
        en_var = S_stats['var'] * mixture_size * concentration**2
        
        if  excitation_dist == 'gaussian':
            return stats.norm(en_mean, np.sqrt(en_var))
        elif  excitation_dist == 'log-normal':
            return lognorm_mean_var(en_mean, en_var)
        else:
            raise ValueError("Unknown excitation distribution `%s`. Supported "
                             "are ['gaussian', 'log-normal']" % excitation_dist)
    
    
    @property
    def threshold_factor_compensated(self):
        """ returns the threshold factor corrected for the excitation that was
        actually measured """
        alpha = self.threshold_factor
        return alpha * (self.Nr - 1) / (self.Nr - alpha)
            
            
    @property
    def threshold_factor_numerics(self):
        """ returns the threshold factor corrected for the excitation that was
        actually measured """
        if self.parameters['compensated_threshold']:
            return self.threshold_factor_compensated
        else:
            return self.threshold_factor
            
            
    def excitation_threshold(self, normalized=False):
        """ returns the average excitation threshold that receptors have to
        overcome to be part of the activation pattern.
        `compensated` determines whether the compensated threshold factor is
            used to determine the excitation threshold
        """
        en_stats = self.excitation_statistics(normalized=normalized)
        return self.threshold_factor_numerics * en_stats['mean']

        
    def excitation_threshold_statistics_new(self, normalized=False):
        """ returns the statistics of the excitation threshold that receptors
        have to overcome to be part of the activation pattern. """
        alpha = self.threshold_factor_numerics
        en_stats = self.excitation_statistics(normalized=normalized)
        
        #en_thresh_var = alpha**2 * en_stats['var'] #/ self.Nr
        
        return {'mean': alpha * en_stats['mean'],
                'std': alpha * en_stats['std'] / np.sqrt(self.Nr),
                'var': alpha**2 * en_stats['var'] / self.Nr }
        
        
    def excitation_threshold_statistics(self, normalized=False):
        """ returns the statistics of the excitation threshold that receptors
        have to overcome to be part of the activation pattern. """
        alpha = self.threshold_factor_numerics

        # get statistics of the individual concentrations
        c_stats = self.concentration_statistics(normalized=normalized)
        c2_mean = c_stats['mean']**2 + c_stats['var']
        c2_mean_sum = c2_mean.sum()
         
        # get statistics of the total concentration c_tot = \sum_i c_i
        if normalized:
            ctot_mean = 1
        else:
            ctot_mean = c_stats['mean'].sum()
        ctot_var = c_stats['var'].sum()
         
        # get statistics of the sensitivities S_ni
        S_stats = self.sensitivity_stats()
        S_mean = S_stats['mean']
         
        # calculate statistics of the sum e_n = \sum_i S_ni * c_i        
        en_mean_mean = S_mean * ctot_mean
        en_mean_var  = S_mean**2 * ctot_var \
                       + S_stats['var'] * c2_mean_sum / self.Nr
         
        # return the statistics of the excitation threshold
        en_thresh_var = alpha**2 * en_mean_var
        return {'mean': alpha * en_mean_mean,
                'var': en_thresh_var,
                'std': np.sqrt(en_thresh_var)}


    def activity_distance_uncorrelated(self, mixture_size=1):
        """ calculate the expected difference (Hamming distance) between the
        activity pattern of two completely uncorrelated mixtures of size
        `mixture_size`. The mixture size influences the statistics of the
        excitations and thus influences the result slightly.
        """
        e_thresh = (self.threshold_factor_numerics
                    * self.mean_sensitivity
                    * mixture_size)
        en_dist = self.excitation_distribution_mixture(mixture_size=mixture_size)
        p_a = en_dist.sf(e_thresh)
        return 2 * self.Nr * p_a * (1 - p_a)
            
          
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
            return self.activity_distance_uncorrelated()
        
        # determine the excitation thresholds
        en_dist = self.excitation_distribution_mixture()
        e_thresh_0 = self.threshold_factor_numerics * en_dist.mean()
        e_thresh_rho = (1 + c_ratio) * e_thresh_0
        p_inact = en_dist.cdf(e_thresh_0)
        
        # determine the probability of changing the activity of a receptor
        def integrand(e1):
            """ integrand for the activation probability """ 
            cdf_val = en_dist.cdf((e_thresh_rho - e1) / c_ratio)
            return cdf_val * en_dist.pdf(e1)
            
        p_on = p_inact - integrate.quad(integrand, en_dist.a, e_thresh_0)[0]
        p_off = integrate.quad(integrand, e_thresh_0, en_dist.b)[0]
    
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
            return self.activity_distance_uncorrelated(mixture_size=mixture_size)
    
        s = mixture_size
        sB = mixture_overlap
        sD = s - sB # number of different ligands
        
        if not self.is_homogeneous_mixture:
            logging.warn('Activity distances can only be estimated for '
                         'homogeneous mixtures, where all ligands have the '
                         'same concentration distribution. We are thus using '
                         'the means of the concentration means and variances.')

        #c_mean = self.c_means.mean()
        #c_var = self.c_vars.mean()
        S_stats = self.sensitivity_stats()
        en_mean = S_stats['mean'] #* c_mean
        en_var = S_stats['var'] #(S_stats['mean']**2 + S_stats['var']) #* c_var
    
        # determine the excitation thresholds
        e_thresh_total = s * self.threshold_factor_numerics * en_mean

        # get the excitation distributions of different mixture sizes
        en_dist_total = lognorm_mean_var(s*en_mean, s*en_var)
        en_dist_same = lognorm_mean_var(sB*en_mean, sB*en_var)
        en_dist_diff = lognorm_mean_var(sD*en_mean, sD*en_var)

        # determine the probability of changing the activity of a receptor
        # use the general definition of the integral
        def p_change_xor(e_same):
            """ probability that the different ligands of either mixture
            bring the excitation above threshold """ 
            # prob that the excitation does not exceed threshold
            cdf_val = en_dist_diff.cdf(e_thresh_total - e_same) 
            return cdf_val * (1 - cdf_val) * en_dist_same.pdf(e_same)
    
        # integrate over all excitations of the common ligands
        p_different = integrate.quad(p_change_xor, en_dist_total.a,
                                     e_thresh_total)[0]
    
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


    def receptor_activity(self, normalized_variables=True, integrate=False):
        """ return the probability with which a single receptor is activated 
        by typical mixtures """
        if normalized_variables:
            en_thresh = self.threshold_factor_numerics * self.mean_sensitivity
            en_dist = self.excitation_distribution(normalized=True)
            
        else:
            en_thresh = self.excitation_threshold()
            en_dist = self.excitation_distribution(normalized=False)
            
        if integrate:
            # probability that the excitation exceeds the threshold
            en_thresh_stats = self.excitation_threshold_statistics_new(
                                                normalized=normalized_variables)
            en_thresh_dist = lognorm_mean_var(en_thresh_stats['mean'],
                                              en_thresh_stats['var'])
            
            return en_dist.expect(en_thresh_dist.cdf)
            #return en_thresh_dist.expect(en_dist.sf)
            
        else: 
            # probability that excitation exceeds the deterministic threshold
            return en_dist.sf(en_thresh)
            

    def receptor_crosstalk(self):
        """ calculates the average activity of the receptor as a response to 
        single ligands. """
        raise NotImplementedError


    def mutual_information(self, normalized_variables=True, integrate=False):
        """ calculates the typical mutual information """
        logging.warn('The estimate of the mutual information does not include '
                     'receptor correlations, yet.')
        a_n = self.receptor_activity(normalized_variables, integrate)
        MI = -self.Nr * (xlog2x(a_n) + xlog2x(1 - a_n))
        return MI
            
            
            
class AdaptiveThresholdTheoryReceptorFactors(AdaptiveThresholdMixin,
                                             LibrarySparseBase):
    """ class for theoretical calculations where all sensitivities are drawn
    independently from log-normal distributions. Here each receptor can have a
    scaling factor to bias the sensitivities. """

    parameters_default = {
        'excitation_distribution': 'log-normal', 
    }
    
            
    def __init__(self, num_substrates, num_receptors, mean_sensitivity=1,
                 width=1, receptor_factors=None, parameters=None):
        
        super(AdaptiveThresholdTheoryReceptorFactors, self).__init__(
                                     num_substrates, num_receptors, parameters)
        
        if receptor_factors is None:
            receptor_factors = np.ones(num_receptors, np.double)
        elif np.isscalar(receptor_factors):
            receptor_factors = np.full(num_receptors, receptor_factors,
                                       np.double)
        
        self.mean_sensitivity = mean_sensitivity
        self.width = width
        self.receptor_factors = receptor_factors
        
        
    @property
    def repr_params(self):
        """ return the important parameters that are shown in __repr__ """
        params = super(AdaptiveThresholdTheoryReceptorFactors, self).repr_params
        params.append('<S>=%g' % self.mean_sensitivity)
        params.append('width=%g' % self.width)
        params.append('factors=%g +- %g' % (self.receptor_factors.mean(),
                                            self.receptor_factors.std()))
        return params
    
    
    @classmethod
    def get_random_arguments(cls, mean_sensitivity=None, width=None,
                             receptor_factors=None, **kwargs):
        """ create random args for creating test instances """
        args = super(AdaptiveThresholdMixin, cls).get_random_arguments(**kwargs)
        
        if mean_sensitivity is None:
            mean_sensitivity = 0.5 * np.random.random()
        if width is None:
            width = 0.5 * np.random.random()
        if receptor_factors is None:
            receptor_factors = 0.5 * np.random.random(args['num_receptors'])
            
        args['parameters']['mean_sensitivity'] = mean_sensitivity
        args['parameters']['width'] = width
        args['parameters']['receptor_factors'] = receptor_factors

        return args
    
    
    def sensitivity_stats(self, with_receptor_factors=True):
        """ returns statistics of the sensitivity distribution """
        if with_receptor_factors:
            factors = self.receptor_factors
        else:
            factors = 1
        S0 = factors * self.mean_sensitivity
        var = factors**2 * S0**2 * (np.exp(self.width**2) - 1)
        return {'mean': S0, 'std': np.sqrt(var), 'var': var}

    
    def excitation_statistics(self, with_receptor_factors=True):
        """ calculates the statistics of the excitation of the receptors.
        Returns the expected mean excitation, the variance, and the covariance
        matrix of any given receptor """
        if self.is_correlated_mixture:
            raise NotImplementedError('Not implemented for correlated mixtures')

        # get statistics of the individual concentrations
        c_stats = self.concentration_statistics()
        c2_mean = c_stats['mean']**2 + c_stats['var']
        c2_mean_sum = c2_mean.sum()

        # get statistics of the total concentration c_tot = \sum_i c_i
        ctot_mean = c_stats['mean'].sum()
        ctot_var = c_stats['var'].sum()
        
        # get statistics of the sensitivities S_ni
        S_stats = self.sensitivity_stats(with_receptor_factors)
        S_mean = S_stats['mean']
        
        # calculate statistics of the sum e_n = \sum_i S_ni * c_i        
        en_mean = S_mean * ctot_mean
        en_var  = S_mean**2 * ctot_var + S_stats['var'] * c2_mean_sum

        return {'mean': en_mean, 'std': np.sqrt(en_var), 'var': en_var}
        

    def excitation_distribution_basic(self):
        """ returns a scipy.stats distribution for the excitations with the
        given mean and standard deviation, ignoring the receptor_factors
        """ 
        excitation_dist = self.parameters['excitation_distribution']
        en_stats = self.excitation_statistics(with_receptor_factors=False)
        
        if  excitation_dist == 'gaussian':
            return stats.norm(en_stats['mean'], en_stats['std'])
        elif  excitation_dist == 'log-normal':
            return lognorm_mean_var(en_stats['mean'], en_stats['var'])
        else:
            raise ValueError("Unknown excitation distribution `%s`. Supported "
                             "are ['gaussian', 'log-normal']" % excitation_dist)   
            

    def excitation_distributions(self):
        """ return a list of scipy.stats distribution for the excitations with
        the given mean and standard deviation, including the receptor_factors 
        """ 
        excitation_dist = self.parameters['excitation_distribution']
        en_stats = self.excitation_statistics(with_receptor_factors=True)
        
        if  excitation_dist == 'gaussian':
            return [stats.norm(mean, std)
                    for mean, std in zip(en_stats['mean'], en_stats['std'])]
        elif  excitation_dist == 'log-normal':
            return [lognorm_mean_var(mean, var)
                    for mean, var in zip(en_stats['mean'], en_stats['var'])]
        else:
            raise ValueError("Unknown excitation distribution `%s`. Supported "
                             "are ['gaussian', 'log-normal']" % excitation_dist)    
    
        
    def receptor_activity(self):
        """ return the probability with which a single receptor is activated 
        by typical mixtures """
        en_dist = self.excitation_distribution_basic()

        # calculate the effective excitation threshold per receptor 
        alpha = self.threshold_factor
        factors = self.receptor_factors
            
        en_threshs = (alpha / (self.Nr - alpha)
                      * (factors.sum() / factors - 1)
                      * en_dist.mean()) 
            
        # probability that excitation exceeds the deterministic threshold
        a_n = [en_dist.sf(en_thresh) for en_thresh in en_threshs]
         
        return a_n        
            
            
    def mutual_information(self):
        """ calculates the typical mutual information """
        logging.warn('The estimate of the mutual information does not include '
                     'receptor correlations, yet.')
        MI = sum(-(xlog2x(a_n) + xlog2x(1 - a_n))
                 for a_n in self.receptor_activity())
        return MI        