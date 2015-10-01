'''
Created on May 1, 2015

@author: zwicker
'''

from __future__ import division, absolute_import

import numpy as np
from scipy import linalg, special
from six.moves import range

from .lib_gau_base import LibraryGaussianBase
from ..library_numeric_base import (LibraryNumericMixin, get_sensitivity_matrix,
                                    optimize_continuous_library)


class LibraryGaussianNumeric(LibraryNumericMixin, LibraryGaussianBase):
    """ represents a single receptor library that handles continuous mixtures,
    which are defined by their concentration mean and variance """

    # default parameters that are used to initialize a class if not overwritten
    parameters_default = {
        'max_num_receptors': 28,           #< prevents memory overflows
        'positive_concentrations': False,  #< ensure positive concentrations?
        'sensitivity_matrix': None,        #< default sensitivity matrix
        'sensitivity_matrix_params': None, #< parameters determining S_ni
        'monte_carlo_steps': 'auto',       #< default steps for monte carlo
        'monte_carlo_steps_min': 1e4,      #< minimal steps for monte carlo
        'monte_carlo_steps_max': 1e5,      #< maximal steps for monte carlo
    }
    
            
    @classmethod
    def create_test_instance(cls, **kwargs):
        """ creates a test instance used for consistency tests """
        obj = super(LibraryGaussianNumeric, cls).create_test_instance(**kwargs)

        # determine optimal parameters for the interaction matrix
#         from binary_response.gaussian_mixtures.lib_gau_theory import LibraryContinuousLogNormal
#         theory = LibraryContinuousLogNormal.from_other(obj)
#         obj.choose_sensitivity_matrix(**theory.get_optimal_library())
        return obj
    

    @property
    def _sample_steps(self):
        """ returns the number of steps that are sampled """
        if self.parameters['monte_carlo_steps'] == 'auto':
            steps_min = self.parameters['monte_carlo_steps_min']
            steps_max = self.parameters['monte_carlo_steps_max']
            steps = np.clip(10 * 2**self.Nr, steps_min, steps_max) 
            # Here, the factor 10 is an arbitrary scaling factor
        else:
            steps = self.parameters['monte_carlo_steps']
            
        return int(steps)
    
        
    def _sample_mixtures(self, steps=None):
        """ sample mixtures with uniform probability yielding single mixtures """
            
        if steps is None:
            steps = self._sample_steps
            
        positive_concentrations = self.parameters['positive_concentrations']
        
        if self.is_correlated_mixture:
            # yield correlated Gaussian variables
            
            ci_means = self.concentrations

            # pre-calculations
            Lij = linalg.cholesky(self.covariance, lower=True)
            
            for _ in range(steps):
                c = np.dot(Lij, np.random.randn(self.Ns)) + ci_means
                if positive_concentrations:
                    yield np.maximum(c, 0)
                else:
                    yield c
        
        else:
            # yield independent Gaussian variables
            ci_stats = LibraryGaussianBase.concentration_statistics(self)
            ci_mean = ci_stats['mean']
            ci_std = ci_stats['std']
            
            for _ in range(steps):
                c = np.random.randn(self.Ns) * ci_std + ci_mean
                if positive_concentrations:
                    yield np.maximum(c, 0)
                else:
                    yield c


    def choose_sensitivity_matrix(self, distribution, mean_sensitivity=1,
                                  **kwargs):
        """ chooses the sensitivity matrix """
        self.sens_mat, sens_mat_params = get_sensitivity_matrix(
                  self.Nr, self.Ns, distribution, mean_sensitivity, **kwargs)

        # save the parameters determining this matrix
        self.parameters['sensitivity_matrix_params'] = sens_mat_params

    choose_sensitivity_matrix.__doc__ = get_sensitivity_matrix.__doc__  


    def concentration_statistics(self, method='auto', **kwargs):
        """ returns statistics for each individual substrate """
        if method == 'auto':
            method = 'monte_carlo'

        if method == 'estimate':            
            return self.concentration_statistics_estimate(**kwargs)
        elif method == 'monte_carlo' or method == 'monte-carlo':
            return self.concentration_statistics_monte_carlo(**kwargs)
        else:
            raise ValueError('Unknown method `%s`.' % method)

    
    def concentration_statistics_estimate(self, approx_covariance=True):
        """ returns statistics for each individual substrate """
        # get the statistics of the unrestricted case
        stats_unres = LibraryGaussianBase.concentration_statistics(self)
        
        if not self.parameters['positive_concentrations']:
            # simple case where concentrations are unrestricted
            return stats_unres

        # calculate the effect of restricting the concentration to positive
        u_mean = stats_unres['mean']
        u_var = stats_unres['var']

        # prepare some constants        
        PI2 = 2 * np.pi
        e_arg = u_mean / np.sqrt(2*u_var)
        e_erf = special.erf(e_arg)
        
        # calculate the mean of the censored distribution
        ci_mean = (0.5 * u_mean * (1 + e_erf)
                   + np.sqrt(u_var/PI2) * np.exp(-e_arg**2))
 
        # calculate the variance of the censored distribution
        t1 = -u_var * np.exp(-e_arg) / PI2
        t2 = -u_mean * e_erf * np.sqrt(u_var/PI2) * np.exp(-e_arg**2)
        t3 = 0.5 * (1 + e_erf) * (u_var + 0.5 * u_mean**2 * (1 - e_erf))
        ci_var = t1 + t2 + t3

        result = {'mean': ci_mean, 'std': np.sqrt(ci_var), 'var': ci_var}

        if self.is_correlated_mixture:
            if approx_covariance:
                factor = np.sqrt(ci_var / u_var)
                result['cov'] = np.einsum('i,ij,j->ij', factor,
                                          stats_unres['cov'], factor)
            else:
                raise NotImplementedError('Estimation of the covariance is not '
                                          'implemented. An approximation is '
                                          'returned if approx_covariance=True')
        else:
            result['cov'] = np.diag(ci_var)
            result['cov_is_diagonal'] = True
        
        # return the results in a dictionary to be able to extend it later
        return result


    def optimize_library(self, target, **kwargs):
        """ optimizes the current library to maximize the result of the target
        function using gradient descent. By default, the function returns the
        best value and the associated sensitivity matrix as result.        
        """
        return optimize_continuous_library(self, target, **kwargs)
        
