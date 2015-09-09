'''
Created on May 1, 2015

@author: zwicker
'''

from __future__ import division

import logging

import numpy as np

from utils.math_distributions import lognorm_mean, loguniform_mean
from utils.misc import is_pos_semidef
from .library_base import LibrarySparseBase  # @UnresolvedImport
from ..binary_mixtures.library_numeric import _sample_binary_mixtures



class LibrarySparseNumeric(LibrarySparseBase):
    """ represents a single receptor library that handles sparse mixtures """

    # default parameters that are used to initialize a class if not overwritten
    parameters_default = {
        'max_num_receptors': 28,           #< prevents memory overflows
        'max_steps': 1e7,                  #< maximal number of steps 
        'interaction_matrix': None,        #< will be calculated if not given
        'interaction_matrix_params': None, #< parameters determining I_ai
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
            if self.parameters['interaction_matrix'] is not None:
                initialize_state = 'exact'
            elif self.parameters['interaction_matrix_params'] is not None:
                initialize_state = 'ensemble'
            else:
                initialize_state = 'zero'
        
        # initialize the state using the chosen protocol
        if initialize_state is None or initialize_state == 'zero':
            self.int_mat = np.zeros((self.Nr, self.Ns), np.double)
            
        elif initialize_state == 'exact':
            # initialize the state using saved parameters
            int_mat = self.parameters['interaction_matrix']
            if int_mat is None:
                logging.warn('Interaction matrix was not given. Initialize '
                             'empty matrix.')
                self.int_mat = np.zeros((self.Nr, self.Ns), np.double)
            else:
                self.int_mat = int_mat.copy()
            
        elif initialize_state == 'ensemble':
            # initialize the state using the ensemble parameters
                params = self.parameters['interaction_matrix_params']
                if params is None:
                    logging.warn('Parameters for interaction matrix were not '
                                 'specified. Initialize empty matrix.')
                    self.int_mat = np.zeros((self.Nr, self.Ns), np.double)
                else:
                    self.choose_interaction_matrix(**params)
            
        else:
            raise ValueError('Unknown initialization protocol `%s`' % 
                             initialize_state)
            
        assert self.int_mat.shape == (self.Nr, self.Ns)
         
            
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
        from .library_theory import LibrarySparseBinary  
        theory = LibrarySparseBinary.from_other(obj_base) 
        
        obj = cls.from_other(obj_base)
        obj.choose_interaction_matrix(**theory.get_optimal_library())
        return obj
    

    def choose_interaction_matrix(self, distribution, mean_sensitivity=1,
                                  ensure_mean=False, **kwargs):
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
        
        int_mat_params = {'distribution': distribution,
                          'mean_sensitivity': mean_sensitivity,
                          'ensure_mean': ensure_mean}

        if distribution == 'const':
            # simple constant matrix
            self.int_mat = np.full(shape, mean_sensitivity)

        elif distribution == 'binary':
            # choose a binary matrix with a typical scale
            if 'standard_deviation' in kwargs:
                standard_deviation = kwargs.pop('standard_deviation')
                S_mean2 = mean_sensitivity ** 2
                density = S_mean2 / (S_mean2 + standard_deviation**2)
            elif 'density' in kwargs:
                density = kwargs.pop('density')
                standard_deviation = mean_sensitivity * np.sqrt(1/density - 1)
            else:
                standard_deviation = 1
                S_mean2 = mean_sensitivity ** 2
                density = S_mean2 / (S_mean2 + standard_deviation**2)

            if density > 1:
                raise ValueError('Standard deviation is too large.')
                
            int_mat_params['standard_deviation'] = standard_deviation
            
            if density == 0:
                # simple case of empty matrix
                self.int_mat = np.zeros(shape)
                
            elif density >= 1:
                # simple case of full matrix
                self.int_mat = np.full(shape, mean_sensitivity)
                
            else:
                # choose receptor substrate interaction randomly and don't worry
                # about correlations
                S_scale = mean_sensitivity / density
                nonzeros = (np.random.random(shape) < density)
                self.int_mat = S_scale * nonzeros 

        elif distribution == 'log_normal':
            # log normal distribution
            if 'standard_deviation' in kwargs:
                standard_deviation = kwargs.pop('standard_deviation')
                cv = standard_deviation / mean_sensitivity 
                spread = np.sqrt(np.log(cv**2 + 1))
            elif 'spread' in kwargs:
                spread = kwargs.pop('spread')
                cv = np.sqrt(np.exp(spread**2) - 1)
                standard_deviation = mean_sensitivity * cv
            else:
                standard_deviation = 1
                cv = standard_deviation / mean_sensitivity
                spread = np.sqrt(np.log(cv**2 + 1))

            correlation = kwargs.pop('correlation', 0)
            int_mat_params['standard_deviation'] = standard_deviation
            int_mat_params['correlation'] = correlation

            if spread == 0 and correlation == 0:
                # edge case without randomness
                self.int_mat = np.full(shape, mean_sensitivity)

            elif correlation != 0:
                # correlated receptors
                mu = np.log(mean_sensitivity) - 0.5 * spread**2
                mean = np.full(self.Nr, mu)
                cov = np.full((self.Nr, self.Nr), correlation * spread**2)
                np.fill_diagonal(cov, spread**2)
                vals = np.random.multivariate_normal(mean, cov, size=self.Ns).T
                self.int_mat = np.exp(vals)

            else:
                # uncorrelated receptors
                dist = lognorm_mean(mean_sensitivity, spread)
                self.int_mat = dist.rvs(shape)
                
        elif distribution == 'log_uniform':
            # log uniform distribution
            spread = kwargs.pop('spread', 1)
            int_mat_params['spread'] = spread

            if spread == 0:
                self.int_mat = np.full(shape, mean_sensitivity)
            else:
                dist = loguniform_mean(mean_sensitivity, np.exp(spread))
                self.int_mat = dist.rvs(shape)
            
        elif distribution == 'log_gamma':
            raise NotImplementedError
            
        elif distribution == 'normal':
            # normal distribution
            spread = kwargs.pop('spread', 1)
            correlation = kwargs.pop('correlation', 0)
            int_mat_params['spread'] = spread
            int_mat_params['correlation'] = correlation

            if spread == 0 and correlation == 0:
                # edge case without randomness
                self.int_mat = np.full(shape, mean_sensitivity)
                
            elif correlation != 0:
                # correlated receptors
                mean = np.full(self.Nr, mean_sensitivity)
                cov = np.full((self.Nr, self.Nr), correlation * spread**2)
                np.fill_diagonal(cov, spread**2)
                if not is_pos_semidef(cov):
                    raise ValueError('The specified correlation leads to a '
                                     'correlation matrix that is not positive '
                                     'semi-definite.')
                vals = np.random.multivariate_normal(mean, cov, size=self.Ns)
                self.int_mat = vals.T

            else:
                # uncorrelated receptors
                self.int_mat = np.random.normal(loc=mean_sensitivity,
                                                scale=spread,
                                                size=shape)
            
        elif distribution == 'gamma':
            raise NotImplementedError
            
        else:
            raise ValueError('Unknown distribution `%s`' % distribution)
            
        if ensure_mean:
            self.int_mat *= mean_sensitivity / self.int_mat.mean()
            
        # save the parameters determining this matrix
        self.parameters['interaction_matrix_params'] = int_mat_params
        
        # raise an error if keyword arguments have not been used
        if len(kwargs) > 0:
            raise ValueError('The following keyword arguments have not been '
                             'used: %s' % str(kwargs)) 


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

    
    def concentration_statistics(self):
        """ returns statistics for each individual substrate """
        if self.is_correlated_mixture:
            return self.concentration_statistics_monte_carlo()
        else:
            return super(LibrarySparseNumeric, self).concentration_statistics()


    def concentration_statistics_monte_carlo(self):
        """ calculates mixture statistics using a metropolis algorithm """
        count = 0
        hist1d = np.zeros(self.Ns)
        hist2d = np.zeros((self.Ns, self.Ns))

        # sample mixtures uniformly
        # FIXME: use better online algorithm that is less prone to canceling
        for c in self._sample_mixtures():
            count += 1
            hist1d += c
            hist2d += np.outer(c, c)
        
        # calculate the frequency and the correlations 
        ci_mean = hist1d/count
        cij_corr = hist2d/count - np.outer(ci_mean, ci_mean)
        
        c_vars = np.diag(cij_corr)
        return {'mean': ci_mean, 'std': np.sqrt(c_vars), 'var': c_vars,
                'cov': cij_corr}
    

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
                        
                            
    def excitation_statistics_monte_carlo(self, ret_correlations=False):
        """
        calculates the statistics of the excitation of the receptors.
        Returns the mean excitation, the variance, and the covariance matrix.
        
        The algorithms used here have been taken from
            https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance
        """
        # prevent integer overflow in collecting activity patterns
        assert self.Nr <= self.parameters['max_num_receptors'] <= 63

        S_ni = self.int_mat

        if ret_correlations:
            # calculate the mean and the covariance matrix
    
            # prepare variables holding the necessary data
            en_mean = np.zeros(self.Nr)
            enm_cov = np.zeros((self.Nr, self.Nr))
            
            # sample mixtures and safe the requested data
            for count, c_i in enumerate(self._sample_mixtures(), 1):
                e_n = np.dot(S_ni, c_i)
                delta = (e_n - en_mean) / count
                en_mean += delta
                enm_cov += (count - 1) * np.outer(delta, delta) - enm_cov / count
                
            # calculate the requested statistics
            if count < 2:
                enm_cov.fill(np.nan)
            else:
                enm_cov *= count / (count - 1)
            
            en_var = np.diag(enm_cov)
            
            return {'mean': en_mean, 'std': np.sqrt(en_var), 'var': en_var,
                    'cov': enm_cov}
            
        else:
            # only calculate the mean and the variance
    
            # prepare variables holding the necessary data
            en_mean = np.zeros(self.Nr)
            en_square = np.zeros(self.Nr)
            
            # sample mixtures and safe the requested data
            for count, c_i in enumerate(self._sample_mixtures(), 1):
                e_n = np.dot(S_ni, c_i)
                delta = e_n - en_mean
                en_mean += delta / count
                en_square += delta*(e_n - en_mean)
                
            # calculate the requested statistics
            if count < 2:
                en_var.fill(np.nan)
            else:
                en_var = en_square / (count - 1)
    
            return {'mean': en_mean, 'std': np.sqrt(en_var), 'var': en_var}
            
    
    def excitation_statistics_estimate(self):
        """
        calculates the statistics of the excitation of the receptors.
        Returns the mean excitation, the variance, and the covariance matrix.
        """
        if self.is_correlated_mixture:
            raise NotImplementedError('Not implemented for correlated mixtures')
        
        c_stats = self.concentration_statistics()
        
        # calculate statistics of the sum s_n = S_ni * c_i        
        S_ni = self.int_mat
        en_mean = np.dot(S_ni, c_stats['mean'])
        enm_cov = np.einsum('ni,mi,i->nm', S_ni, S_ni, c_stats['var'])
        en_var = np.diag(enm_cov)
        
        return {'mean': en_mean, 'std': np.sqrt(en_var), 'var': en_var,
                'cov': enm_cov}
            
        
    def receptor_activity(self, method='auto', ret_correlations=False, **kwargs):
        """ calculates the average activity of each receptor
        
        `method` can be one of [monte_carlo', 'estimate'].
        """
        if method == 'auto':
            method = 'monte_carlo'
                
        if method == 'monte_carlo' or method == 'monte-carlo':
            return self.receptor_activity_monte_carlo(ret_correlations, **kwargs)
        
        elif method == 'estimate':
            return self.receptor_activity_estimate(ret_correlations, **kwargs)
        
        else:
            raise ValueError('Unknown method `%s`.' % method)
                        

    def receptor_activity_monte_carlo(self, ret_correlations=False):
        """ calculates the average activity of each receptor """ 
        # prevent integer overflow in collecting activity patterns
        assert self.Nr <= self.parameters['max_num_receptors'] <= 63

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
                                   approx_prob=False, clip=False):
        """ estimates the average activity of each receptor """
        en_stats = self.excitation_statistics_estimate()

        # calculate the receptor activity
        r_n = self._estimate_qn_from_en(en_stats, approx_prob=approx_prob)
        if clip:
            np.clip(r_n, 0, 1, r_n)

        if ret_correlations:
            # calculate the correlated activity 
            q_nm = self._estimate_qnm_from_en(en_stats)
            r_nm = r_n**2 + q_nm
            if clip:
                np.clip(r_nm, 0, 1, r_nm)

            return r_n, r_nm
        else:
            return r_n
               
 
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
        
                                   
    def mutual_information_monte_carlo(self, ret_prob_activity=False):
        """ calculate the mutual information using a monte carlo strategy. The
        number of steps is given by the model parameter 'monte_carlo_steps' """
        # prevent integer overflow in collecting activity patterns
        assert self.Nr <= self.parameters['max_num_receptors'] <= 63

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
        q_n = count_a / count_a.sum()
        
        # calculate the mutual information from the result pattern
        MI = -sum(q*np.log2(q) for q in q_n if q != 0)

        if ret_prob_activity:
            return MI, q_n
        else:
            return MI

                    
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
        MI = self._estimate_mutual_information_from_q_values(
                                           q_n, q_nm, use_polynom=use_polynom)
        
        if ret_prob_activity:
            return MI, q_n
        else:
            return MI
        
            