'''
Created on May 1, 2015

@author: zwicker
'''

from __future__ import division

import functools
import logging
import time

import numpy as np
from scipy import optimize

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
            logging.debug('Initialize sensitivity matrix to zero.')
            self.sens_mat = np.zeros((self.Nr, self.Ns), np.double)
            
        elif initialize_state == 'exact':
            # initialize the state using saved parameters
            sens_mat = self.parameters['sensitivity_matrix']
            if sens_mat is None:
                logging.warn('Sensitivity matrix was not given. Initialize '
                             'zero matrix.')
                self.sens_mat = np.zeros((self.Nr, self.Ns), np.double)
            else:
                logging.debug('Initialize with given sensitivity matrix.')
                self.sens_mat = sens_mat.copy()
            
        elif initialize_state == 'ensemble':
            # initialize the state using the ensemble parameters
            params = self.parameters['sensitivity_matrix_params']
            if params is None:
                logging.warn('Parameters for sensitivity matrix were not '
                             'specified. Initialize zero matrix.')
                self.sens_mat = np.zeros((self.Nr, self.Ns), np.double)
            else:
                logging.debug('Choose sensitivity matrix from given '
                              'parameters')
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

        # determine optimal parameters for the sensitivity matrix
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

        
    def concentration_statistics_estimate(self):
        """ estimate the statistics for each individual substrate """
        return super(LibrarySparseNumeric, self).concentration_statistics()
        
        
    def optimize_library(self, target, direction='max', steps=100,
                         method='cma', ret_info=False, args=None, verbose=False):
        """ optimizes the current library to maximize the result of the target
        function using gradient descent. By default, the function returns the
        best value and the associated sensitivity matrix as result.        
        """
        # get the target function to call
        target_function = getattr(self, target)
        if args is not None:
            target_function = functools.partial(target_function, **args)

        # define the cost function
        if direction == 'min':
            def cost_function(sens_mat_flat):
                """ cost function to minimize """
                self.sens_mat.flat = sens_mat_flat.flat
                return target_function()
            
        elif direction == 'max':
            def cost_function(sens_mat_flat):
                """ cost function to minimize """
                self.sens_mat.flat = sens_mat_flat.flat
                return -target_function()
            
        else:
            raise ValueError('Unknown optimization direction `%s`' % direction)

        if ret_info:
            # store extra information
            start_time = time.time()
            info = {'values': []}
            
            cost_function_inner = cost_function
            def cost_function(sens_mat_flat):
                """ wrapper function to store calculated costs """
                cost = cost_function_inner(sens_mat_flat)
                info['values'].append(cost)
                return cost
        
        if method == 'cma':
            # use Covariance Matrix Adaptation Evolution Strategy algorithm
            try:
                import cma  # @UnresolvedImport
            except ImportError:
                raise ImportError('The module `cma` is not available. Please '
                                  'install it using `pip install cma` or '
                                  'choose a different optimization method.')
            
            # prepare the arguments for the optimization call    
            x0 = self.sens_mat.flat
            sigma = 0.5 * np.mean(x0) #< initial step size
            options = {'maxfevals': steps,
                       'bounds': [0, np.inf],
                       'verb_disp': 100 * int(verbose),
                       'verb_log': 0}
            
            # call the optimizer
            res = cma.fmin(cost_function, x0, sigma, options=options)
            
            # get the result
            state_best = res[0].reshape((self.Nr, self.Ns))
            value_best = res[1]
            if ret_info: 
                info['states_considered'] = res[3]
                info['iterations'] = res[4]
            
        else:
            # use the standard scipy function
            res = optimize.minimize(cost_function, self.sens_mat.flat,
                                    method=method, options={'maxiter': steps})
            value_best =  res.fun
            state_best = res.x.reshape((self.Nr, self.Ns))
            if ret_info: 
                info['states_considered'] = res.nfev
                info['iterations'] = res.nit
            
        if direction == 'max':
            value_best *= -1
        
        self.sens_mat = state_best.copy()

        if ret_info:
            info['total_time'] = time.time() - start_time    
            info['performance'] = info['states_considered'] / info['total_time']
            return value_best, state_best, info
        else:
            return value_best, state_best
        
        
           
    