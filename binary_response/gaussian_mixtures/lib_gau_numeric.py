'''
Created on May 1, 2015

@author: zwicker
'''

from __future__ import division

import logging

import numpy as np
from scipy import linalg, special
from six.moves import range

from .lib_gau_base import LibraryGaussianBase
from ..library_base import LibraryNumericMixin
from utils.math_distributions import lognorm_mean


class LibraryGaussianNumeric(LibraryGaussianBase, LibraryNumericMixin):
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
    

    def __init__(self, num_substrates, num_receptors, parameters=None):
        """ initialize a receptor library by setting the number of receptors,
        the number of substrates it can respond to, and optional additional
        parameters in the parameter dictionary """
        # the call to the inherited method also sets the default parameters from
        # this class
        super(LibraryGaussianNumeric, self).__init__(num_substrates,
                                                     num_receptors,
                                                     parameters)        

        # prevent integer overflow in collecting activity patterns
        assert num_receptors <= self.parameters['max_num_receptors'] <= 63

        initialize_state = self.parameters['initialize_state']
        
        if initialize_state == 'auto': 
            # use exact values if saved or ensemble properties otherwise
            if self.parameters['sensitivity_matrix'] is not None:
                initialize_state = 'exact'
            elif self.parameters['sensitivity_matrix_params'] is not None:
                initialize_state = 'ensemble'
            else:
                initialize_state = 'zero'
        
        # initialize the state using the chosen protocol
        if initialize_state is None or initialize_state == 'zero':
            self.sens_mat = np.zeros((self.Nr, self.Ns), np.double)
            
        elif initialize_state == 'exact':
            # initialize the state using saved parameters
            sens_mat = self.parameters['sensitivity_matrix']
            if sens_mat is None:
                logging.warn('Interaction matrix was not given. Initialize '
                             'empty matrix.')
                self.sens_mat = np.zeros((self.Nr, self.Ns), np.double)
            else:
                self.sens_mat = sens_mat.copy()
            
        elif initialize_state == 'ensemble':
            # initialize the state using the ensemble parameters
                params = self.parameters['sensitivity_matrix_params']
                if params is None:
                    logging.warn('Parameters for interaction matrix were not '
                                 'specified. Initialize empty matrix.')
                    self.sens_mat = np.zeros((self.Nr, self.Ns), np.double)
                else:
                    self.choose_sensitivity_matrix(**params)
            
        else:
            raise ValueError('Unknown initialization protocol `%s`' % 
                             initialize_state)
            
        assert self.sens_mat.shape == (self.Nr, self.Ns)
            
            
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
            parent = super(LibraryGaussianNumeric, self)
            ci_stats = parent.concentration_statistics()
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
        """ creates a interaction matrix with the given properties
            `distribution` determines the distribution from which we choose the
                entries of the sensitivity matrix
            `mean_sensitivity` should in principle set the mean sensitivity,
                although there are some exceptional distributions. For instance,
                for binary distributions `mean_sensitivity` sets the
                magnitude of the entries that are non-zero.
            Some distributions might accept additional parameters.
        """
        shape = (self.Nr, self.Ns)

        assert mean_sensitivity > 0 

        if distribution == 'const':
            # simple constant matrix
            self.sens_mat = np.full(shape, mean_sensitivity)

        elif distribution == 'binary':
            # choose a binary matrix with a typical scale
            kwargs.setdefault('density', 0)
            if kwargs['density'] == 0:
                # simple case of empty matrix
                self.sens_mat = np.zeros(shape)
            elif kwargs['density'] >= 1:
                # simple case of full matrix
                self.sens_mat = np.full(shape, mean_sensitivity)
            else:
                # choose receptor substrate interaction randomly and don't worry
                # about correlations
                self.sens_mat = (mean_sensitivity * 
                                (np.random.random(shape) < kwargs['density']))

        elif distribution == 'log_normal':
            # log normal distribution
            kwargs.setdefault('sigma', 1)
            if kwargs['sigma'] == 0:
                self.sens_mat = np.full(shape, mean_sensitivity)
            else:
                dist = lognorm_mean(mean_sensitivity, kwargs['sigma'])
                self.sens_mat = dist.rvs(shape)
                
        elif distribution == 'log_uniform':
            raise NotImplementedError
            
        elif distribution == 'log_gamma':
            raise NotImplementedError
            
        elif distribution == 'gamma':
            raise NotImplementedError
            
        else:
            raise ValueError('Unknown distribution `%s`' % distribution)
            
        # save the parameters determining this matrix
        sens_mat_params = {'distribution': distribution,
                          'mean_sensitivity': mean_sensitivity}
        sens_mat_params.update(kwargs)
        self.parameters['sensitivity_matrix_params'] = sens_mat_params 


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

    
    def concentration_statistics_estimate(self, approx_covariance=False):
        """ returns statistics for each individual substrate """
        # get the statistics of the unrestricted case
        parent = super(LibraryGaussianNumeric, self)
        stats_unres = parent.concentration_statistics()
        
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

        ci_std = np.sqrt(ci_var)

        if self.is_correlated_mixture:
            if approx_covariance:
                factor = np.sqrt(ci_var / u_var)
                cij_cov = np.einsum('i,ij,j->ij', factor, stats_unres['cov'],
                                    factor)
            else:
                raise NotImplementedError('Estimation of the covariance is not '
                                          'implemented. An approximation is '
                                          'returned if approx_covariance=True')
        else:
            cij_cov = np.diag(ci_var)
        
        # return the results in a dictionary to be able to extend it later
        return {'mean': ci_mean, 'std': ci_std, 'var': ci_var, 'cov': cij_cov}


    def excitation_statistics(self, method='auto', ret_correlations=True):
        """ calculates the statistics of the excitation of the receptors.
        Returns the mean excitation, the variance, and the covariance matrix.

        `method` can be one of [monte_carlo', 'estimate'].
        """
        if method == 'auto':
            method = 'monte_carlo'
                
        if method == 'monte_carlo' or method == 'monte-carlo':
            return self.excitation_statistics_monte_carlo(ret_correlations)
        elif method == 'estimate':
            return self.excitation_statistics_estimate(ret_correlations)
        else:
            raise ValueError('Unknown method `%s`.' % method)
                            
    
    def excitation_statistics_estimate(self, ret_correlations=True):
        """
        calculates the statistics of the excitation of the receptors.
        Returns the mean excitation, the variance, and the covariance matrix.
        """
        c_stats = self.concentration_statistics_estimate(approx_covariance=True)
        
        # calculate statistics of the sum s_n = S_ni * c_i        
        S_ni = self.sens_mat
        en_mean = np.dot(S_ni, c_stats['mean'])
        enm_cov = np.einsum('ni,mj,ij->nm', S_ni, S_ni, c_stats['cov'])
        en_var = np.diag(enm_cov)
        
        return {'mean': en_mean, 'std': np.sqrt(en_var), 'var': en_var,
                'cov': enm_cov}


    def receptor_activity(self, ret_correlations=False):
        """ calculates the average activity of each receptor """ 
        return self.receptor_activity_monte_carlo(ret_correlations)


    def mutual_information(self, ret_prob_activity=False):
        """ calculate the mutual information using a monte carlo strategy. The
        number of steps is given by the model parameter 'monte_carlo_steps' """
        return self.mutual_information_monte_carlo(ret_prob_activity)
    
    