'''
Created on Sep 10, 2015

@author: David Zwicker <dzwicker@seas.harvard.edu>
'''

from __future__ import division

import functools
import time

import numpy as np
from scipy import optimize

from utils.math_distributions import lognorm_mean, loguniform_mean
from utils.misc import is_pos_semidef



class LibraryNumericMixin(object):
    """ Mixin class that defines functions that are useful for numerical 
    calculations.
    
    The iteration over mixtures must be implemented in the subclass. Here, we 
    expect the methods `_sample_mixtures` and `_sample_steps` to exist, which
    are a generator of mixtures and its expected length, respectively.    
    """
    
    def concentration_statistics(self, method='auto', **kwargs):
        """ calculates mixture statistics using a metropolis algorithm
        Returns the mean concentration, the variance, and the covariance matrix.

        `method` can be one of [monte_carlo', 'estimate'].
        """
        if method == 'auto':
            method = 'monte_carlo'
                
        if method == 'monte_carlo' or method == 'monte-carlo':
            return self.concenration_statistics_monte_carlo(**kwargs)
        elif method == 'estimate':
            return self.concentration_statistics_estimate(**kwargs)
        else:
            raise ValueError('Unknown method `%s`.' % method)
            
            
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
        
        ci_var = np.diag(cij_corr)
        return {'mean': ci_mean, 'std': np.sqrt(ci_var), 'var': ci_var,
                'cov': cij_corr}


    def excitation_statistics(self, method='auto', ret_correlations=True,
                              **kwargs):
        """ calculates the statistics of the excitation of the receptors.
        Returns the mean excitation, the variance, and the covariance matrix.

        `method` can be one of [monte_carlo', 'estimate'].
        """
        if method == 'auto':
            method = 'monte_carlo'
                
        if method == 'monte_carlo' or method == 'monte-carlo':
            return self.excitation_statistics_monte_carlo(ret_correlations,
                                                          **kwargs)
        elif method == 'estimate':
            return self.excitation_statistics_estimate(**kwargs)
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

        S_ni = self.sens_mat

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
                            
    
    def excitation_statistics_estimate(self, **kwargs):
        """
        calculates the statistics of the excitation of the receptors.
        Returns the mean excitation, the variance, and the covariance matrix.
        """
        c_stats = self.concentration_statistics_estimate(**kwargs)
        
        # calculate statistics of e_n = \sum_i S_ni * c_i        
        S_ni = self.sens_mat
        en_mean = np.dot(S_ni, c_stats['mean'])
        cov_is_diagonal = c_stats.get('cov_is_diagonal', False)
        if cov_is_diagonal:
            enm_cov = np.einsum('ni,mi,i->nm', S_ni, S_ni, c_stats['var'])
        else:
            enm_cov = np.einsum('ni,mj,ij->nm', S_ni, S_ni, c_stats['cov'])
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

        S_ni = self.sens_mat

        r_n = np.zeros(self.Nr)
        if ret_correlations:
            r_nm = np.zeros((self.Nr, self.Nr))
        
        for c_i in self._sample_mixtures():
            e_n = np.dot(S_ni, c_i)
            a_n = (e_n >= 1)
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
                                   excitation_model='default', clip=False):
        """ estimates the average activity of each receptor """
        en_stats = self.excitation_statistics_estimate()

        # calculate the receptor activity
        r_n = self._estimate_qn_from_en(en_stats,
                                        excitation_model=excitation_model)
        if clip:
            np.clip(r_n, 0, 1, r_n)

        if ret_correlations:
            # calculate the correlated activity 
            q_nm = self._estimate_qnm_from_en(en_stats)
            r_nm = np.outer(r_n, r_n) + q_nm
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
                                    excitation_model='default', clip=False):
        """ calculates the average activity of the receptor as a response to 
        single ligands. """
        en_stats = self.excitation_statistics_estimate()

        # calculate the receptor crosstalk
        q_nm = self._estimate_qnm_from_en(en_stats)
        if clip:
            np.clip(q_nm, 0, 1, q_nm)

        if ret_receptor_activity:
            # calculate the receptor activity
            q_n = self._estimate_qn_from_en(en_stats, excitation_model)
            if clip:
                np.clip(q_n, 0, 1, q_n)

            return q_n, q_nm
        else:
            return q_nm        
        
        
    def mutual_information(self, excitation_method='auto', ret_prob_activity=False,
                           **kwargs):
        """ calculate the mutual information of the receptor array.

        `excitation_method` can be one of [monte_carlo', 'estimate'].
        """
        if excitation_method == 'auto':
            excitation_method = 'monte_carlo'
                
        if excitation_method == 'monte_carlo' or excitation_method == 'monte-carlo':
            return self.mutual_information_monte_carlo(ret_prob_activity)
        elif excitation_method == 'estimate':
            return self.mutual_information_estimate(ret_prob_activity, **kwargs)
        else:
            raise ValueError('Unknown excitation_method `%s`.' % excitation_method)

                                                   
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
            a = (np.dot(self.sens_mat, c) >= 1)
            
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

                    
    def mutual_information_estimate(self, ret_prob_activity=False,
                                    excitation_model='default',
                                    mutual_information_method='default',
                                    clip=True):
        """ returns a simple estimate of the mutual information.
        `clip` determines whether the approximated probabilities should be
            clipped to [0, 1] before being used to calculate the mutual info.
        """
        q_n, q_nm = self.receptor_crosstalk_estimate(
            ret_receptor_activity=True, \
            excitation_model=excitation_model,
            clip=clip
        )
        
        # calculate the approximate mutual information
        MI = self._estimate_MI_from_q_values(
                                    q_n, q_nm, method=mutual_information_method)
        
        if clip:
            MI = np.clip(MI, 0, self.Nr)
        
        if ret_prob_activity:
            return MI, q_n
        else:
            return MI
        

            
def get_sensitivity_matrix(Nr, Ns, distribution, mean_sensitivity=1,
                           ensure_mean=False, ret_params=True, **kwargs):
    """ creates a sensitivity matrix with the given properties
        `Nr` is the number of receptors
        `Ns` is the number of substrates/ligands
        `distribution` determines the distribution from which we choose the
            entries of the sensitivity matrix
        `mean_sensitivity` should in principle set the mean sensitivity,
            although there are some exceptional distributions. For instance,
            for binary distributions `mean_sensitivity` sets the
            magnitude of the entries that are non-zero.
        `ensure_mean` makes sure that the mean of the matrix is indeed equal to
            `mean_sensitivity`
        `ret_params` determines whether a dictionary with the parameters that
            lead to the calculated sensitivity is also returned 
        Some distributions might accept additional parameters.
    """
    shape = (Nr, Ns)

    assert mean_sensitivity > 0
    
    sens_mat_params = {'distribution': distribution,
                      'mean_sensitivity': mean_sensitivity,
                      'ensure_mean': ensure_mean}

    if distribution == 'const':
        # simple constant matrix
        sens_mat = np.full(shape, mean_sensitivity)

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
            
        sens_mat_params['standard_deviation'] = standard_deviation
        
        if density == 0:
            # simple case of empty matrix
            sens_mat = np.zeros(shape)
            
        elif density >= 1:
            # simple case of full matrix
            sens_mat = np.full(shape, mean_sensitivity)
            
        else:
            # choose receptor substrate interaction randomly and don't worry
            # about correlations
            S_scale = mean_sensitivity / density
            nonzeros = (np.random.random(shape) < density)
            sens_mat = S_scale * nonzeros 

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
        sens_mat_params['standard_deviation'] = standard_deviation
        sens_mat_params['correlation'] = correlation

        if spread == 0 and correlation == 0:
            # edge case without randomness
            sens_mat = np.full(shape, mean_sensitivity)

        elif correlation != 0:
            # correlated receptors
            mu = np.log(mean_sensitivity) - 0.5 * spread**2
            mean = np.full(Nr, mu)
            cov = np.full((Nr, Nr), correlation * spread**2)
            np.fill_diagonal(cov, spread**2)
            vals = np.random.multivariate_normal(mean, cov, size=Ns).T
            sens_mat = np.exp(vals)

        else:
            # uncorrelated receptors
            dist = lognorm_mean(mean_sensitivity, spread)
            sens_mat = dist.rvs(shape)
            
    elif distribution == 'log_uniform':
        # log uniform distribution
        spread = kwargs.pop('spread', 1)
        sens_mat_params['spread'] = spread

        if spread == 0:
            sens_mat = np.full(shape, mean_sensitivity)
        else:
            dist = loguniform_mean(mean_sensitivity, np.exp(spread))
            sens_mat = dist.rvs(shape)
        
    elif distribution == 'log_gamma':
        raise NotImplementedError
        
    elif distribution == 'normal':
        # normal distribution
        spread = kwargs.pop('spread', 1)
        correlation = kwargs.pop('correlation', 0)
        sens_mat_params['spread'] = spread
        sens_mat_params['correlation'] = correlation

        if spread == 0 and correlation == 0:
            # edge case without randomness
            sens_mat = np.full(shape, mean_sensitivity)
            
        elif correlation != 0:
            # correlated receptors
            mean = np.full(Nr, mean_sensitivity)
            cov = np.full((Nr, Nr), correlation * spread**2)
            np.fill_diagonal(cov, spread**2)
            if not is_pos_semidef(cov):
                raise ValueError('The specified correlation leads to a '
                                 'correlation matrix that is not positive '
                                 'semi-definite.')
            vals = np.random.multivariate_normal(mean, cov, size=Ns)
            sens_mat = vals.T

        else:
            # uncorrelated receptors
            sens_mat = np.random.normal(loc=mean_sensitivity, scale=spread,
                                        size=shape)

    elif distribution == 'uniform':
        # uniform sensitivity distribution
        S_min = kwargs.pop('S_min', 0)
        S_max = 2 * mean_sensitivity - S_min
        
        # choose random sensitivities
        sens_mat = np.random.uniform(S_min, S_max, size=shape)
        
    elif distribution == 'gamma':
        raise NotImplementedError
        
    else:
        raise ValueError('Unknown distribution `%s`' % distribution)
        
    if ensure_mean:
        sens_mat *= mean_sensitivity / sens_mat.mean()

    # raise an error if keyword arguments have not been used
    if len(kwargs) > 0:
        raise ValueError('The following keyword arguments have not been '
                         'used: %s' % str(kwargs)) 
    
    if ret_params:    
        # return the parameters determining this sensitivity matrix
        return sens_mat, sens_mat_params
    else:
        return sens_mat
    


def optimize_continuous_library(model, target, direction='max', steps=100,
                                method='cma', ret_info=False, args=None,
                                verbose=False):
    """ optimizes the current library to maximize the result of the target
    function using gradient descent. By default, the function returns the
    best value and the associated sensitivity matrix as result.        
    """
    # get the target function to call
    target_function = getattr(model, target)
    if args is not None:
        target_function = functools.partial(target_function, **args)

    # define the cost function
    if direction == 'min':
        def cost_function(sens_mat_flat):
            """ cost function to minimize """
            model.sens_mat.flat = sens_mat_flat.flat
            return target_function()
        
    elif direction == 'max':
        def cost_function(sens_mat_flat):
            """ cost function to minimize """
            model.sens_mat.flat = sens_mat_flat.flat
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
        x0 = model.sens_mat.flat
        sigma = 0.5 * np.mean(x0) #< initial step size
        options = {'maxfevals': steps,
                   'bounds': [0, np.inf],
                   'verb_disp': 100 * int(verbose),
                   'verb_log': 0}
        
        # call the optimizer
        res = cma.fmin(cost_function, x0, sigma, options=options)
        
        # get the result
        state_best = res[0].reshape((model.Nr, model.Ns))
        value_best = res[1]
        if ret_info: 
            info['states_considered'] = res[3]
            info['iterations'] = res[4]
        
    else:
        # use the standard scipy function
        res = optimize.minimize(cost_function, model.sens_mat.flat,
                                method=method, options={'maxiter': steps})
        value_best =  res.fun
        state_best = res.x.reshape((model.Nr, model.Ns))
        if ret_info: 
            info['states_considered'] = res.nfev
            info['iterations'] = res.nit
        
    if direction == 'max':
        value_best *= -1
    
    model.sens_mat = state_best.copy()

    if ret_info:
        info['total_time'] = time.time() - start_time    
        info['performance'] = info['states_considered'] / info['total_time']
        return value_best, state_best, info
    else:
        return value_best, state_best
    