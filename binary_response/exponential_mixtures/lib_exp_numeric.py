'''
Created on May 1, 2015

@author: zwicker
'''

from __future__ import division

import numpy as np
from six.moves import range

from .lib_exp_base import LibraryExponentialBase
from ..library_numeric_base import LibraryNumericMixin, get_sensitivity_matrix



class LibraryExponentialNumeric(LibraryNumericMixin, LibraryExponentialBase):
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
    
    