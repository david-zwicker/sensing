'''
Created on May 1, 2015

@author: zwicker
'''

from __future__ import division

import numpy as np
from scipy import stats, special
from six.moves import range

from .library_base import LibrarySparseBase  # @UnresolvedImport


LN2 = np.log(2)


class LibrarySparseNumeric(LibrarySparseBase):
    """ represents a single receptor library that handles sparse mixtures """

    # default parameters that are used to initialize a class if not overwritten
    parameters_default = {
        'max_num_receptors': 28,           #< prevents memory overflows
        'max_steps': 1e7,               #< maximal number of steps 
        'interaction_matrix': None,        #< will be calculated if not given
        'interaction_matrix_params': None, #< parameters determining I_ai
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

        # prevent integer overflow in collecting activity patterns
        assert num_receptors <= self.parameters['max_num_receptors'] <= 63

        initialize_state = self.parameters['initialize_state'] 
        int_mat_shape = (self.Nr, self.Ns)
        
        if initialize_state is None:
            # do not initialize with anything
            self.int_mat = np.zeros(int_mat_shape, np.uint8)
            
        elif initialize_state == 'exact':
            # initialize the state using saved parameters
                self.int_mat = self.parameters['interaction_matrix'].copy()
            
        elif initialize_state == 'ensemble':
            # initialize the state using the ensemble parameters
                params = self.parameters['interaction_matrix_params']
                self.choose_interaction_matrix(**params)
            
        elif initialize_state == 'auto':
            # use exact values if saved or ensemble properties otherwise
            if self.parameters['interaction_matrix'] is not None:
                # copy the given matrix
                self.int_mat = self.parameters['interaction_matrix'].copy()
            elif self.parameters['interaction_matrix_params'] is not None:
                # create a matrix with the given properties
                params = self.parameters['interaction_matrix_params']
                self.choose_interaction_matrix(**params)
            else:
                # initialize the interaction matrix with zeros
                self.int_mat = np.zeros(int_mat_shape, np.uint8)

        else:
            raise ValueError('Unknown initialization protocol `%s`' % 
                             initialize_state)

        assert self.int_mat.shape == int_mat_shape
         
            
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
        obj = super(LibrarySparseNumeric, cls).create_test_instance(**kwargs)

        # determine optimal parameters for the interaction matrix
        from .library_theory import LibrarySparseBinary  # @UnresolvedImport
        theory = LibrarySparseBinary.from_other(obj)
        obj.choose_interaction_matrix(**theory.get_optimal_library())
        return obj
    

    def choose_interaction_matrix(self, distribution, typical_sensitivity=1,
                                  **kwargs):
        """ creates a interaction matrix with the given properties
            `distribution` determines the distribution from which we choose the
                entries of the sensitivity matrix
            `typical_sensitivity` should in principle set the mean sensitivity,
                although there are some exceptional distributions. For instance,
                for binary distributions `typical_sensitivity` sets the
                magnitude of the entries that are non-zero.
            Some distributions might accept additional parameters.
        """
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
            kwargs.setdefault('sigma', 1)
            if kwargs['sigma'] == 0:
                self.int_mat = np.full(shape, typical_sensitivity)
            else:
                sigma = kwargs['sigma']
                mu = typical_sensitivity * np.exp(0.5*sigma**2)
                dist = stats.lognorm(scale=mu, s=sigma)
                self.int_mat = dist.rvs(shape)
                
        elif distribution == 'log_uniform':
            raise NotImplementedError
            
        elif distribution == 'log_gamma':
            raise NotImplementedError
            
        elif distribution == 'normal':
            # normal distribution
            kwargs.setdefault('sigma', 1)
            if kwargs['sigma'] == 0:
                self.int_mat = np.full(shape, typical_sensitivity)
            else:
                self.int_mat = np.random.normal(loc=typical_sensitivity,
                                                scale=kwargs['sigma'],
                                                size=shape)
            
        elif distribution == 'gamma':
            raise NotImplementedError
            
        else:
            raise ValueError('Unknown distribution `%s`' % distribution)
            
        # save the parameters determining this matrix
        int_mat_params = {'distribution': distribution,
                          'typical_sensitivity': typical_sensitivity}
        int_mat_params.update(kwargs)
        self.parameters['interaction_matrix_params'] = int_mat_params 


    @property
    def _sample_steps(self):
        """ returns the number of steps that are sampled """
        return self.monte_carlo_steps


    def _sample_mixtures(self):
        """ sample mixtures with uniform probability yielding single mixtures """
        # use simple monte carlo algorithm
        p_i = self.substrate_probabilities
        d_i = self.concentrations
        
        for _ in range(self._sample_steps):
            # choose ligands that are _not_ in a mixture
            b_not = (np.random.random(self.Ns) >= p_i)

            # choose a mixture vector according to substrate probabilities
            c = np.random.exponential(size=self.Ns) * d_i
            c[b_not] = 0
            yield c


    def receptor_activity(self, method='auto', ret_correlations=False):
        """ calculates the average activity of each receptor
        
        `method` can be one of [monte_carlo', 'estimate'].
        """
        if method == 'auto':
            method = 'monte_carlo'
                
        if method == 'monte_carlo' or method == 'monte-carlo':
            return self.receptor_activity_monte_carlo(ret_correlations)
        
        elif method == 'estimate':
            return self.receptor_activity_estimate(ret_correlations)
        
        else:
            raise ValueError('Unknown method `%s`.' % method)
                        

    def receptor_activity_monte_carlo(self, ret_correlations=False):
        """ calculates the average activity of each receptor """ 
        if self.correlated_mixture:
            raise NotImplementedError('Not implemented for correlated mixtures')

        S_ni = self.int_mat

        r_n = np.zeros(self.Nr)
        if ret_correlations:
            r_nm = np.zeros((self.Nr, self.Nr))
        
        for c_i in self._sample_mixtures():
            a_n = (np.dot(S_ni, c_i) >= 1)
            r_n[a_n] += 1
            if ret_correlations:
                r_nm[np.outer(a_n, a_n)] += 1
            
        r_n /= self._sample_steps
        if ret_correlations:
            r_nm /= self._sample_steps
            return r_n, r_nm
        else:
            return r_n
    
    
    def receptor_activity_estimate(self, ret_correlations=False,
                                   approx_prob=False):
        """ estimates the average activity of each receptor """ 
        if self.correlated_mixture:
            raise NotImplementedError('Not implemented for correlated mixtures')
        
        S_ni = self.int_mat
        p_i = self.substrate_probabilities
        d_i = self.concentrations
        
        b_mean = np.dot(S_ni, p_i*d_i)
        b_var = np.dot(S_ni**2, p_i * d_i**2)
        b_std = np.sqrt(b_var)

        # handle division by zero correctly
        with np.errstate(divide='ignore'):
            delta = np.divide(b_mean - 1, b_std)  #< deviation from optimum
            # delta will be +- infinity if b_std is zero

        if approx_prob:
            r_n = 0.5 + delta / np.sqrt(2*np.pi)
            np.clip(r_n, 0, 1, r_n)
        else:
            r_n = 0.5 * special.erfc(-delta / np.sqrt(2))
            
        if ret_correlations:
            b_covar = np.dot(S_ni[:, None, :] * S_ni[None, :, :], p_i * d_i**2)
            with np.errstate(divide='ignore', invalid='ignore'):
                rho = np.divide(b_covar, np.outer(b_std, b_std))  #< correlation coefficient
            r_nm = (0.25
                    + np.add.outer(delta, delta) / np.sqrt(8*np.pi)
                    + (np.outer(delta, delta) + rho) / (2*np.pi)
                    )
            # Replace values that are nan with zero. This might not be exact,
            # but only occurs in corner cases that are not interesting to us  
            r_nm[np.isnan(r_nm)] = 0

            np.clip(r_nm, 0, 1, r_nm)
            return r_n, r_nm
        
        else:
            return r_n
               
 
    def receptor_crosstalk(self, method='auto', ret_receptor_activity=False):
        """ calculates the average activity of the receptor as a response to 
        single ligands.
        
        `method` can be ['brute_force', 'monte_carlo', 'estimate', 'auto'].
            If it is 'auto' than the method is chosen automatically based on the
            problem size.
        """
        # calculate receptor crosstalk from the observed probabilities
        r_n, r_nm = self.receptor_activity(method, ret_correlations=True)
        q_nm = r_nm - np.outer(r_n, r_n)
        
        if ret_receptor_activity:
            return r_n, q_nm # q_n = r_n
        else:
            return q_nm 
        
                           
    def mutual_information(self, ret_prob_activity=False):
        """ calculate the mutual information using a monte carlo strategy. The
        number of steps is given by the model parameter 'monte_carlo_steps' """
        if self.correlated_mixture:
            raise NotImplementedError('Not implemented for correlated mixtures')

        base = 2 ** np.arange(0, self.Nr)

        # sample mixtures according to the probabilities of finding
        # substrates
        count_a = np.zeros(2**self.Nr)
        for c in self._sample_mixtures():
            # get the activity vector ...
            a = (np.dot(self.int_mat, c) >= 1)
            
            # ... and represent it as a single integer
            a_id = np.dot(base, a)
            # increment counter for this output
            count_a[a_id] += 1
            
        # count_a contains the number of times output pattern a was observed.
        # We can thus construct P_a(a) from count_a. 
        prob_a = count_a / count_a.sum()
        
        # calculate the mutual information from the result pattern
        MI = -sum(pa*np.log2(pa) for pa in prob_a if pa != 0)

        if ret_prob_activity:
            return MI, prob_a
        else:
            return MI

                    
    def mutual_information_estimate(self, approx_prob=False):
        """ returns a simple estimate of the mutual information.
        `approx_prob` determines whether the probabilities of encountering
            substrates in mixtures are calculated exactly or only approximative,
            which should work for small probabilities. """

        # this might be not the right approach
        q_n = self.receptor_activity_estimate(approx_prob=approx_prob)
        q_nm = self.receptor_crosstalk_estimate(approx_prob=approx_prob)
                    
        # calculate the approximate mutual information
        MI = self.Nr
        MI -= 0.5/LN2 * np.sum((2*q_n - 1)**2)
        MI -= 8/LN2 * np.sum(np.triu(q_nm, 1)**2)
        
        return MI
            