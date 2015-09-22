'''
Created on May 1, 2015

@author: zwicker
'''

from __future__ import division

import logging

import numpy as np
from six.moves import range

from .lib_exp_base import LibraryExponentialBase
from ..library_numeric_base import LibraryNumericMixin, get_sensitivity_matrix



class LibraryExponentialNumeric(LibraryExponentialBase, LibraryNumericMixin):
    """ represents a single receptor library that handles continuous mixtures,
    which are defined by their concentration mean and variance """

    # default parameters that are used to initialize a class if not overwritten
    parameters_default = {
        'max_num_receptors': 28,    #< prevents memory overflows
        'sensitivity_matrix': None, #< will be calculated if not given
        'sensitivity_matrix_params': None, #< parameters determining I_ai
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
        super(LibraryExponentialNumeric, self).__init__(num_substrates,
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
        obj = super(LibraryExponentialNumeric, cls).create_test_instance(**kwargs)

        # determine optimal parameters for the interaction matrix
        from .lib_exp_theory import LibraryExponentialLogNormal
        theory = LibraryExponentialLogNormal.from_other(obj)
        obj.choose_sensitivity_matrix(**theory.get_optimal_library())
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
        
        c_means = self.concentration_means
        
        for _ in range(steps):
            # choose a mixture vector according to substrate probabilities
            yield np.random.exponential(size=self.Ns) * c_means


    def choose_sensitivity_matrix(self, distribution, mean_sensitivity=1,
                                  **kwargs):
        """ chooses the sensitivity matrix """
        self.sens_mat, sens_mat_params = get_sensitivity_matrix(
                  self.Nr, self.Ns, distribution, mean_sensitivity, **kwargs)

        # save the parameters determining this matrix
        self.parameters['sensitivity_matrix_params'] = sens_mat_params

    choose_sensitivity_matrix.__doc__ = get_sensitivity_matrix.__doc__  


    def receptor_activity(self, ret_correlations=False):
        """ calculates the average activity of each receptor """ 
        return self.receptor_activity_monte_carlo(ret_correlations)

    
    def mutual_information(self, ret_prob_activity=False):
        """ calculate the mutual information using a monte carlo strategy. The
        number of steps is given by the model parameter 'monte_carlo_steps' """
        return self.mutual_information_monte_carlo(ret_prob_activity)
    
    