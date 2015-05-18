'''
Created on May 1, 2015

@author: zwicker
'''

from __future__ import division

import numpy as np
from scipy import stats
from six.moves import range

from .library_base import LibraryContinuousBase


class LibraryContinuousNumeric(LibraryContinuousBase):
    """ represents a single receptor library that handles continuous mixtures """

    # default parameters that are used to initialize a class if not overwritten
    parameters_default = {
        'max_num_receptors': 28,    #< prevents memory overflows
        'interaction_matrix': None, #< will be calculated if not given
        'interaction_matrix_params': None, #< parameters determining I_ai
        'monte_carlo_steps': 1e5,   #< default number of monte carlo steps
    }
    

    def __init__(self, num_substrates, num_receptors, parameters=None):
        """ initialize a receptor library by setting the number of receptors,
        the number of substrates it can respond to, and optional additional
        parameters in the parameter dictionary """
        # the call to the inherited method also sets the default parameters from
        # this class
        super(LibraryContinuousNumeric, self).__init__(num_substrates,
                                                       num_receptors,
                                                       parameters)        

        # prevent integer overflow in collecting activity patterns
        assert num_receptors <= self.parameters['max_num_receptors'] <= 63

        int_mat_shape = (self.Nr, self.Ns)
        if self.parameters['interaction_matrix'] is not None:
            # copy the given matrix
            self.int_mat[:] = self.parameters['interaction_matrix']
            assert self.int_mat.shape == int_mat_shape
            
        elif self.parameters['interaction_matrix_params'] is not None:
            # create a matrix with the given properties
            params = self.parameters['interaction_matrix_params']
            self.choose_interaction_matrix(**params)
            
        else:
            # initialize the interaction matrix with zeros
            self.int_mat = np.zeros(int_mat_shape, np.uint8)
            
            
    @classmethod
    def create_test_instance(cls, **kwargs):
        """ creates a test instance used for consistency tests """
        obj = super(LibraryContinuousNumeric, cls).create_test_instance()

        # determine optimal parameters for the interaction matrix
        from .library_theory import LibraryContinuousLogNormal
        theory = LibraryContinuousLogNormal.from_other(obj)
        obj.choose_interaction_matrix(**theory.get_optimal_library())
        return obj


    def choose_interaction_matrix(self, distribution, typical_sensitivity=1,
                                  **kwargs):
        """ creates a interaction matrix with the given properties """
        shape = (self.Nr, self.Ns)

        assert typical_sensitivity > 0 

        if distribution == 'const':
            # simple constant matrix
            self.int_mat = np.full(shape, typical_sensitivity)

        elif distribution == 'binary':
            # choose a binary matrix with a typical scale
            kwargs.setdefault('density', 0)
            if kwargs['density'] == 0:
                # simple case of empty matrix
                self.int_mat = np.zeros(shape)
            elif kwargs['density'] >= 1:
                # simple case of full matrix
                self.int_mat = np.full(shape, typical_sensitivity)
            else:
                # choose receptor substrate interaction randomly and don't worry
                # about correlations
                self.int_mat = (typical_sensitivity * 
                                (np.random.random(shape) < kwargs['density']))

        elif distribution == 'log_normal':
            # log normal distribution
            kwargs.setdefault('sigma', 0.1)
            if kwargs['sigma'] == 0:
                self.int_mat = np.full(shape, typical_sensitivity)
            else:
                dist = stats.lognorm(scale=typical_sensitivity,
                                     s=kwargs['sigma'])
                self.int_mat = dist.rvs(shape)
                
        elif distribution == 'log_uniform':
            raise NotImplementedError
            
        elif distribution == 'log_gamma':
            raise NotImplementedError
            
        elif distribution == 'gamma':
            raise NotImplementedError
            
        else:
            raise ValueError('Unknown distribution `%s`' % distribution)
            
        # save the parameters determining this matrix
        int_mat_params = {'distribution': distribution,
                          'typical_sensitivity': typical_sensitivity}
        int_mat_params.update(kwargs)
        self.parameters['interaction_matrix_params'] = int_mat_params 


    def activity_single(self):
        """ calculates the average activity of each receptor """ 
        steps = int(self.parameters['monte_carlo_steps'])        
    
        c_means = self.concentration_means
    
        count_a = np.zeros(self.Nr)
        for _ in range(steps):
            # choose a mixture vector according to substrate probabilities
            c = np.random.exponential(size=self.Ns) * c_means
            
            # get the associated output ...
            alpha = (np.dot(self.int_mat, c) >= 1)
            
            count_a[alpha] += 1
            
        return count_a/steps

    
    def mutual_information(self, ret_prob_activity=False):
        """ calculate the mutual information using a monte carlo strategy. The
        number of steps is given by the model parameter 'monte_carlo_steps' """
                
        base = 2 ** np.arange(0, self.Nr)

        steps = int(self.parameters['monte_carlo_steps'])
        
        c_means = self.concentration_means

        # sample mixtures according to the probabilities of finding
        # substrates
        count_a = np.zeros(2**self.Nr)
        for _ in range(steps):
            # choose a mixture vector according to substrate probabilities
            c = np.random.exponential(size=self.Ns) * c_means
            
            # get the associated output ...
            alpha = (np.dot(self.int_mat, c) >= 1)
            
            # ... and represent it as a single integer
            alpha_id = np.dot(base, alpha)
            # increment counter for this output
            count_a[alpha_id] += 1
            
        # count_a contains the number of times output pattern a was observed.
        # We can thus construct P_a(a) from count_a. 
        prob_a = count_a / steps
        
        # calculate the mutual information from the result pattern
        MI = -sum(pa*np.log2(pa) for pa in prob_a if pa != 0)

        if ret_prob_activity:
            return MI, prob_a
        else:
            return MI

            