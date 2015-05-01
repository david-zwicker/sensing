'''
Created on May 1, 2015

@author: zwicker
'''

from __future__ import division

import numpy as np

from .library_base import LibraryContinuousBase


class LibraryContinuousNumeric(LibraryContinuousBase):
    """ represents a single receptor library that handles continuous mixtures """

    # default parameters that are used to initialize a class if not overwritten
    parameters_default = {
        'max_num_receptors': 28,    #< prevents memory overflows
        'interaction_matrix': None, #< will be calculated if not given
        'interaction_matrix_params': None, #< parameters determining I_ai
        'monte_carlo_steps': 100,   #< default number of monte carlo steps
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
            

    def choose_interaction_matrix(self, distribution='log_normal', mean_conc=1,
                                  **kwargs):
        """ creates a interaction matrix with the given properties """
        shape = (self.Nr, self.Ns)

        assert mean_conc > 0 

        if distribution == 'log_normal':
            # log normal distribution
            kwargs.setdefault('spread', 1)
            self.int_mat = np.random.lognormal(mean=np.log(1/mean_conc),
                                               sigma=kwargs['spread'],
                                               size=shape)
            
        else:
            raise ValueError('Unknown distribution `%s`' % distribution)
            
        # save the parameters determining this matrix
        int_mat_params = {'distribution': distribution, 'mean_conc': mean_conc}
        int_mat_params.update(kwargs)
        self.parameters['interaction_matrix_params'] = int_mat_params 

    
    def mutual_information_monte_carlo(self, ret_prob_activity=False):
        """ calculate the mutual information using a monte carlo strategy. The
        number of steps is given by the model parameter 'monte_carlo_steps' """
                
        base = 2 ** np.arange(0, self.Nr)

        steps = int(self.parameters['monte_carlo_steps'])
        
        c_factor = -1/self.commonness

        # sample mixtures according to the probabilities of finding
        # substrates
        count_a = np.zeros(2**self.Nr)
        for _ in xrange(steps):
            # choose a mixture vector according to substrate probabilities
            c = np.random.exponential(size=self.Ns) * c_factor
            
            # get the associated output ...
            a = (np.dot(self.int_mat, c) > 1)
            # ... and represent it as a single integer
            a = np.dot(base, a)
            # increment counter for this output
            count_a[a] += 1
            
        # count_a contains the number of times output pattern a was observed.
        # We can thus construct P_a(a) from count_a. 

        prob_a = count_a / steps
        
        # calculate the mutual information from the result pattern
        MI = -sum(pa*np.log2(pa) for pa in prob_a if pa != 0)

        if ret_prob_activity:
            return MI, prob_a
        else:
            return MI

            