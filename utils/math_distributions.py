'''
Created on Feb 24, 2015

@author: David Zwicker <dzwicker@seas.harvard.edu>
'''

from __future__ import division

import numpy as np
from scipy import stats, special, linalg


def lognorm_mean(mean, sigma):
    """ returns a lognormal distribution parameterized by its mean and a spread
    parameter `sigma` """
    mu = mean * np.exp(-0.5 * sigma**2)
    return stats.lognorm(scale=mu, s=sigma)



def lognorm_mean_var(mean, variance):
    """ returns a lognormal distribution parameterized by its mean and its
    variance. """
    mean2 = mean**2
    scale = mean2/np.sqrt(mean2 + variance)
    sigma = np.sqrt(np.log(1 + variance/mean2))
    return stats.lognorm(scale=scale, s=sigma)



def loguniform_mean(mean, sigma):
    """ returns a loguniform distribution parameterized by its mean and a spread
    parameter `sigma`. The ratio between the maximal value and the minimal value
    is given by sigma**2 """
    scale =  mean * (2*sigma*np.log(sigma)) / (sigma**2 - 1)
    return LogUniformDistribution(scale=scale, s=sigma)



def random_log_uniform(v_min, v_max, size):
    """ returns random variables that a distributed uniformly in log space """
    log_min, log_max = np.log(v_min), np.log(v_max)
    res = np.random.uniform(log_min, log_max, size)
    return np.exp(res)



class DeterministicDistribution_gen(stats.rv_continuous):
    """ deterministic distribution that always returns a given value
    Code copied from
    https://docs.scipy.org/doc/scipy/reference/tutorial/stats.html#making-a-continuous-distribution-i-e-subclassing-rv-continuous
    """
    def _cdf(self, x):
        return np.where(x < 0, 0., 1.)
    
    def _stats(self):
        return 0., 0., 0., 0.


DeterministicDistribution = DeterministicDistribution_gen(
    name='DeterministicDistribution'
)



class LogUniformDistribution_gen(stats.rv_continuous):
    """
    Log-uniform distribution.
    """ 
    
    def _rvs(self, s):
        """ random variates """
        # choose the receptor response characteristics
        return random_log_uniform(1/s, s, self._size)
    
    
    def _pdf(self, x, s):
        """ probability density function """
        s = s[0] #< reset broadcasting
        res = np.zeros_like(x)
        idx = (1 < x*s) & (x < s)
        res[idx] = 1/(x[idx] * np.log(s*s))
        return res         
        
        
    def _cdf(self, x, s): 
        """ cumulative probability function """
        s = s[0] #< reset broadcasting
        res = np.zeros_like(x)
        idx = (1 < x*s) & (x < s)
        log_s = np.log(s)
        res[idx] = (log_s + np.log(x[idx]))/(2 * log_s)
        res[x > s] = 1
        
        return res


    def _ppf(self, q, s):
        """ percent point function (inverse of cdf) """
        s = s[0] #< reset broadcasting
        res = np.zeros_like(q)
        idx = (q > 0)
        res[idx] = s**(2*q[idx] - 1)
        return res
    
    
    def _stats(self, s):
        """ calculates statistics of the distribution """
        mean = (s**2 - 1)/(2*s*np.log(s))
        var = ((s**4 - 1)*np.log(s) - (s**2 - 1)**2)/(4* s**2 *np.log(s)**2)
        return mean, var, None, None

    

LogUniformDistribution = LogUniformDistribution_gen(
    a=0, name='LogUniformDistribution'
)



class HypoExponentialDistribution(object):
    """
    Hypoexponential distribution.
    Unfortunately, the framework supplied by scipy.stats.rv_continuous does not
    support a variable number of parameters and we thus only mimic its
    interface here.
    """
    
    def __init__(self, rates, method='sum'):
        """ initializes the hypoexponential distribution.
        `rates` are the rates of the underlying exponential processes
        `method` determines what method is used for calculating the cdf and can
            be either `sum` or `eigen`        
        """
        if method in {'sum', 'eigen'}:
            self.method = method
        
        # prepare the rates of the system
        self.rates = np.asarray(rates)
        self.alpha = 1 / self.rates
        if np.any(rates <= 0):
            raise ValueError('All rates must be positive')
        if len(np.unique(self.alpha)) != len(self.alpha):
            raise ValueError('The current implementation only supports cases '
                             'where all rates are different from each other.')
        
        # calculate terms that we need later
        with np.errstate(divide='ignore'):
            mat = self.alpha[:, None] / (self.alpha[:, None] - self.alpha[None, :])
        mat[(self.alpha[:, None] - self.alpha[None, :]) == 0] = 1
        self._terms = np.prod(mat, 1)
        
    
    def rvs(self, size):
        """ random variates """
        # choose the receptor response characteristics
        return sum(np.random.exponential(scale=alpha, size=size)
                   for alpha in self.alpha)

    
    def mean(self):
        """ mean of the distribution """
        return self.alpha.sum()
    
    
    def variance(self):
        """ variance of the distribution """
        return (2 * np.sum(self.alpha**2 * self._terms)
                - (self.alpha.sum())**2)
    

    def pdf(self, x):
        """ probability density function """
        if not np.isscalar(x):
            x = np.asarray(x)
            res = np.zeros_like(x)
            nz = (x > 0)
            if np.any(nz):
                if self.method == 'sum':
                    factor = np.exp(-x[nz, None]*self.rates[..., :])/self.rates[..., :]
                    res[nz] = np.sum(self._terms[..., :] * factor, axis=1)
                else:
                    Theta = np.diag(-self.rates, 0) + np.diag(self.rates[:-1], 1)
                    for i in np.flatnonzero(nz):
                        res.flat[i] = 1 - linalg.expm(x.flat[i]*Theta)[0, :].sum()
 
        elif x == 0:
            res = 0
        else:
            if self.method == 'sum':
                factor = np.exp(-x*self.rates)/self.ratesx
                res[nz] = np.sum(self._terms * factor)
            else:
                Theta = np.diag(-self.rates, 0) + np.diag(self.rates[:-1], 1)
                res = 1 - linalg.expm(x*Theta)[0, :].sum()
        return res

    
    def cdf(self, x):
        """ cumulative density function """
        if not np.isscalar(x):
            x = np.asarray(x)
            res = np.zeros_like(x)
            nz = (x > 0)
            if np.any(nz):
                factor = np.exp(-x[nz, None]*self.rates[..., :])
                res = 1 - np.sum(self._terms[..., :] * factor, axis=1)
        elif x == 0:
            res = 0
        else:
            factor = np.exp(-x*self.rates)
            res = 1 - np.sum(self._terms * factor)
        return res           



#===============================================================================
# OLD DISTRIBUTIONS THAT MIGHT NOT BE NEEDED ANYMORE
#===============================================================================



class PartialLogNormDistribution_gen(stats.rv_continuous):
    """
    partial log-normal distribution.
    a fraction `frac` of the distribution follows a log-normal distribution,
    while the remaining fraction `1 - frac` is zero
    
    Similar to the lognorm distribution, this does not support any location
    parameter
    """ 
    
    def _rvs(self, s, frac):
        """ random variates """
        # choose the receptor response characteristics
        res = np.exp(s * np.random.standard_normal(self._size))
        if frac != 1:
            # switch off receptors randomly
            res[np.random.random(self._size) > frac] = 0
        return res
    
    
    def _pdf(self, x, s, frac):
        """ probability density function """
        s, frac = s[0], frac[0] #< reset broadcasting
        return frac / (s*x*np.sqrt(2*np.pi)) * np.exp(-1/2*(np.log(x)/s)**2)         
        
        
    def _cdf(self, x, s, frac): 
        """ cumulative probability function """
        s, frac = s[0], frac[0] #< reset broadcasting
        return 1 + frac*(-0.5 + 0.5*special.erf(np.log(x)/(s*np.sqrt(2))))


    def _ppf(self, q, s, frac):
        """ percent point function (inverse of cdf) """
        s, frac = s[0], frac[0] #< reset broadcasting
        q_scale = (q - (1 - frac)) / frac
        res = np.zeros_like(q)
        idx = (q_scale > 0)
        res[idx] = np.exp(s * special.ndtri(q_scale[idx]))
        return res

PartialLogNormDistribution = PartialLogNormDistribution_gen(
    a=0, name='PartialLogNormDistribution'
)




class PartialLogUniformDistribution_gen(stats.rv_continuous):
    """
    partial log-uniform distribution.
    a fraction `frac` of the distribution follows a log-uniform distribution,
    while the remaining fraction `1 - frac` is zero
    """ 
    
    def _rvs(self, s, frac):
        """ random variates """
        # choose the receptor response characteristics
        res = random_log_uniform(1/s, s, self._size)
        # switch off receptors randomly
        if frac != 1:
            res[np.random.random(self._size) > frac] = 0
        return res
    
    
    def _pdf(self, x, s, frac):
        """ probability density function """
        s, frac = s[0], frac[0] #< reset broadcasting
        res = np.zeros_like(x)
        idx = (1 < x*s) & (x < s)
        res[idx] = frac/(x[idx] * np.log(s*s))
        return res         
        
        
    def _cdf(self, x, s, frac): 
        """ cumulative probability function """
        s, frac = s[0], frac[0] #< reset broadcasting
        res = np.zeros_like(x)
        idx = (1 < x*s) & (x < s)
        log_s = np.log(s)
        res[idx] = (log_s + np.log(x[idx]))/(2 * log_s)
        res[x > s] = 1
        
        return (1 - frac) + frac*res


    def _ppf(self, q, s, frac):
        """ percent point function (inverse of cdf) """
        s, frac = s[0], frac[0] #< reset broadcasting
        q_scale = (q - (1 - frac)) / frac
        res = np.zeros_like(q)
        idx = (q_scale > 0)
        res[idx] = s**(2*q_scale[idx] - 1)
        return res
    

PartialLogUniformDistribution = PartialLogUniformDistribution_gen(
    a=0, name='PartialLogUniformDistribution'
)
