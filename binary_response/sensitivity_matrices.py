'''
Created on Sep 10, 2015

@author: David Zwicker <dzwicker@seas.harvard.edu>
'''

from __future__ import division

import numpy as np

from utils.math_distributions import lognorm_mean, loguniform_mean
from utils.misc import is_pos_semidef


            
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

