'''
Created on Jan 5, 2016

@author: David Zwicker <dzwicker@seas.harvard.edu>
'''

from __future__ import division

import logging

import numpy as np
from scipy import stats

from binary_response.sparse_mixtures.lib_spr_theory import LibrarySparseLogNormal

from .at_base import AdaptiveThresholdMixin
from utils.math_distributions import lognorm_mean_var
from utils.misc import xlog2x



class AdaptiveThresholdTheory(AdaptiveThresholdMixin, LibrarySparseLogNormal):
    """ class for theoretical calculations where all sensitivities are drawn
    from the same distribution """

    parameters_default = {
        'excitation_distribution': 'log-normal', 
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
    
    
    @property
    def threshold_factor_compensated(self):
        """ returns the threshold factor corrected for the excitation that was
        actually measured """
        alpha = self.threshold_factor
        return alpha * (self.Nr - 1) / (self.Nr - alpha)
            
            
    def excitation_threshold(self, compensated=False, normalized=False):
        """ returns the average excitation threshold that receptors have to
        overcome to be part of the activation pattern.
        `compensated` determines whether the compensated threshold factor is
            used to determine the excitation threshold
        """
        if compensated:
            alpha = self.threshold_factor_compensated
        else:
            alpha = self.threshold_factor
        
        en_stats = self.excitation_statistics(normalized=normalized)
        return  alpha * en_stats['mean']

        
    def excitation_threshold_statistics_new(self, compensated=False,
                                            normalized=False):
        """ returns the statistics of the excitation threshold that receptors
        have to overcome to be part of the activation pattern. """
        if compensated:
            alpha = self.threshold_factor_compensated
        else:
            alpha = self.threshold_factor

        en_stats = self.excitation_statistics(normalized=normalized)
        
        #en_thresh_var = alpha**2 * en_stats['var'] #/ self.Nr
        
        return {'mean': alpha * en_stats['mean'],
                'std': alpha * en_stats['std'] / np.sqrt(self.Nr),
                'var': alpha**2 * en_stats['var'] / self.Nr }
        
        
    def excitation_threshold_statistics(self, compensated=False,
                                        normalized=False):
        """ returns the statistics of the excitation threshold that receptors
        have to overcome to be part of the activation pattern. """
        if compensated:
            alpha = self.threshold_factor_compensated
        else:
            alpha = self.threshold_factor

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


    def receptor_activity(self, normalized_variables=True, integrate=False):
        """ return the probability with which a single receptor is activated 
        by typical mixtures """
        if normalized_variables:
            en_thresh = (self.threshold_factor_compensated
                         * self.mean_sensitivity)
            en_dist = self.excitation_distribution(normalized=True)
            
        else:
            en_thresh = self.excitation_threshold(compensated=True)
            en_dist = self.excitation_distribution(normalized=False)
            
        if integrate:
            # probability that the excitation exceeds the threshold
            en_thresh_stats = self.excitation_threshold_statistics_new(
                           compensated=True, normalized=normalized_variables)
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
            
            