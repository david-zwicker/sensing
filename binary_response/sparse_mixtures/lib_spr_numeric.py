'''
Created on May 1, 2015

@author: zwicker
'''

from __future__ import division

import logging

import numpy as np

from .lib_spr_base import LibrarySparseBase  # @UnresolvedImport
from ..binary_mixtures.lib_bin_numeric import _sample_binary_mixtures
from ..library_base import LibraryNumericMixin
from ..sensitivity_matrices import get_sensitivity_matrix



class LibrarySparseNumeric(LibrarySparseBase, LibraryNumericMixin):
    """ represents a single receptor library that handles sparse mixtures """

    # default parameters that are used to initialize a class if not overwritten
    parameters_default = {
        'max_num_receptors': 28,           #< prevents memory overflows
        'max_steps': 1e7,                  #< maximal number of steps 
        'sensitivity_matrix': None,        #< will be calculated if not given
        'sensitivity_matrix_params': None, #< parameters determining I_ai
        'fixed_mixture_size': None,     #< fixed m or None
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
        super(LibrarySparseNumeric, self).__init__(num_substrates,
                                                   num_receptors,
                                                   parameters)        
        return

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
         
            
    @property
    def monte_carlo_steps(self):
        """ calculate the number of monte carlo steps to do """
        if self.parameters['monte_carlo_steps'] == 'auto':
            steps_min = self.parameters['monte_carlo_steps_min']
            steps_max = self.parameters['monte_carlo_steps_max']
            steps = np.clip(10 * 2**self.Nr, steps_min, steps_max) 
            # Here, the factor 10 is an arbitrary scaling factor
        else:
            steps = self.parameters['monte_carlo_steps']
            
        return int(steps)

            
    @classmethod
    def create_test_instance(cls, **kwargs):
        """ creates a test instance used for consistency tests """
        parent = super(LibrarySparseNumeric, cls)
        obj_base = parent.create_test_instance(**kwargs)

        # determine optimal parameters for the interaction matrix
        from binary_response.sparse_mixtures.lib_spr_theory import LibrarySparseBinary  
        theory = LibrarySparseBinary.from_other(obj_base) 
        
        obj = cls.from_other(obj_base)
        obj.choose_sensitivity_matrix(**theory.get_optimal_library())
        return obj
    

    def choose_sensitivity_matrix(self, distribution, mean_sensitivity=1,
                                  **kwargs):
        """ chooses the sensitivity matrix """
        self.sens_mat, sens_mat_params = get_sensitivity_matrix(
                  self.Nr, self.Ns, distribution, mean_sensitivity, **kwargs)

        # save the parameters determining this matrix
        self.parameters['sensitivity_matrix_params'] = sens_mat_params

    choose_sensitivity_matrix.__doc__ = get_sensitivity_matrix.__doc__  


    @property
    def _sample_steps(self):
        """ returns the number of steps that are sampled """
        return self.monte_carlo_steps


    def _sample_mixtures(self, steps=None):
        """ sample mixtures with uniform probability yielding single mixtures """
        
        if steps is None:
            steps = self._sample_steps
        
        d_i = self.concentrations
        
        for b in _sample_binary_mixtures(self, steps=steps, dtype=np.bool):
            # boolean b vector is True for the ligands that are present
            
            # choose concentrations for the ligands
            c = np.random.exponential(size=self.Ns) * d_i
            
            # set concentration of ligands that are not present to zero 
            c[~b] = 0
            
            yield c

    
    def concentration_statistics(self, method='auto'):
        """ returns statistics for each individual substrate """
        if method == 'auto':
            if self.is_correlated_mixture:
                method = 'monte_carlo'
            else:
                method = 'estimate'

        if method == 'estimate':            
            return super(LibrarySparseNumeric, self).concentration_statistics()
        elif method == 'monte_carlo' or method == 'monte-carlo':
            return self.concentration_statistics_monte_carlo()
        else:
            raise ValueError('Unknown method `%s`.' % method)


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
            return self.excitation_statistics_estimate()
        else:
            raise ValueError('Unknown method `%s`.' % method)
                            
    
    def excitation_statistics_estimate(self):
        """
        calculates the statistics of the excitation of the receptors.
        Returns the mean excitation, the variance, and the covariance matrix.
        """
        if self.is_correlated_mixture:
            raise NotImplementedError('Not implemented for correlated mixtures')
        
        c_stats = self.concentration_statistics()
        
        # calculate statistics of the sum s_n = S_ni * c_i        
        S_ni = self.sens_mat
        en_mean = np.dot(S_ni, c_stats['mean'])
        enm_cov = np.einsum('ni,mi,i->nm', S_ni, S_ni, c_stats['var'])
        en_var = np.diag(enm_cov)
        
        return {'mean': en_mean, 'std': np.sqrt(en_var), 'var': en_var,
                'cov': enm_cov}
            
 
    def receptor_crosstalk(self, method='auto', ret_receptor_activity=False,
                           clip=False, **kwargs):
        """ calculates the average activity of the receptor as a response to 
        single ligands.
        
        `method` can be ['brute_force', 'monte_carlo', 'estimate', 'auto'].
            If it is 'auto' than the method is chosen automatically based on the
            problem size.
        """
        if method == 'estimate':
            kwargs['clip'] = False

        # calculate receptor activities with the requested `method`            
        r_n, r_nm = self.receptor_activity(method, ret_correlations=True,
                                           **kwargs)
        
        # calculate receptor crosstalk from the observed probabilities
        q_nm = r_nm - np.outer(r_n, r_n)
        if clip:
            np.clip(q_nm, 0, 1, q_nm)
        
        if ret_receptor_activity:
            return r_n, q_nm # q_n = r_n
        else:
            return q_nm

        
    def receptor_crosstalk_estimate(self, ret_receptor_activity=False,
                                    approx_prob=False, clip=False):
        """ calculates the average activity of the receptor as a response to 
        single ligands. """
        en_stats = self.excitation_statistics_estimate()

        # calculate the receptor crosstalk
        q_nm = self._estimate_qnm_from_en(en_stats)
        if clip:
            np.clip(q_nm, 0, 1, q_nm)

        if ret_receptor_activity:
            # calculate the receptor activity
            q_n = self._estimate_qn_from_en(en_stats, approx_prob=approx_prob)
            if clip:
                np.clip(q_n, 0, 1, q_n)

            return q_n, q_nm
        else:
            return q_nm        
        
        
    def mutual_information(self, method='auto', ret_prob_activity=False,
                           **kwargs):
        """ calculate the mutual information of the receptor array.

        `method` can be one of [monte_carlo', 'estimate'].
        """
        if method == 'auto':
            method = 'monte_carlo'
                
        if method == 'monte_carlo' or method == 'monte-carlo':
            return self.mutual_information_monte_carlo(ret_prob_activity)
        elif method == 'estimate':
            return self.mutual_information_estimate(ret_prob_activity, **kwargs)
        else:
            raise ValueError('Unknown method `%s`.' % method)

                    
    def mutual_information_estimate(self, approx_prob=False, clip=True,
                                    use_polynom=False, ret_prob_activity=False):
        """ returns a simple estimate of the mutual information.
        `approx_prob` determines whether the probabilities of encountering
            substrates in mixtures are calculated exactly or only approximative,
            which should work for small probabilities.
        `clip` determines whether the approximated probabilities should be
            clipped to [0, 1] before being used to calculate the mutual info.
        """
        q_n, q_nm = self.receptor_crosstalk_estimate(ret_receptor_activity=True,
                                                     approx_prob=approx_prob,
                                                     clip=clip)
        
        # calculate the approximate mutual information
        MI = self._estimate_MI_from_q_values(
                                           q_n, q_nm, use_polynom=use_polynom)
        
        if ret_prob_activity:
            return MI, q_n
        else:
            return MI
        
            

        
            