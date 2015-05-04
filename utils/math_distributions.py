'''
Created on Feb 24, 2015

@author: David Zwicker <dzwicker@seas.harvard.edu>
'''

from __future__ import division

import numpy as np
from scipy import stats, special



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
    

LogUniformDistribution = LogUniformDistribution_gen(
    a=0, name='LogUniformDistribution'
)




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
