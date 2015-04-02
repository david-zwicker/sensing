'''
Created on Apr 1, 2015

@author: David Zwicker <dzwicker@seas.harvard.edu>
'''

from __future__ import division

import numpy as np
import scipy.misc



class ReceptorLibraryBase(object):
    """ represents a single receptor library. This is a base class that provides
    general functionality and parameter management """


    def __init__(self, num_substrates, num_receptors, hs=None, frac=1):
        """ initialize the receptor library by setting the number of receptors,
        the number of substrates it can respond to, the weights `hs` of the 
        substrates, and the fraction `frac` of substrates a single receptor
        responds to """
        self.Ns = num_substrates
        self.Nr = num_receptors
        if hs is None:
            self.hs = np.zeros(self.Ns) 
        else:
            assert len(hs) == self.Ns
            self.hs = np.array(hs)
        self.frac = frac

            
    def mixture_size_distribution(self):
        """ calculates the probabilities of finding a mixture with a given
        number of components. Returns an array of length Ns + 1 of probabilities
        for finding mixtures with the number of components given by the index
        into the array """
        # calculate the probability of seeing each substrate independently
        prob_h = np.exp(self.hs)/(1 + np.exp(self.hs))
        res = np.zeros(self.Ns + 1)
        res[0] = 1
        for k, p in enumerate(prob_h, 1):
#             r = res[:k].copy()
#             res[:k] *= 1 - p  #< substrate not in the mixture 
#             res[1:k+1] += r*p #< substrate in the mixture
            
            res[k] = res[k-1]*p
            res[1:k] = (1 - p)*res[1:k] + res[:k-1]*p
            res[0] = (1 - p)*res[0]
            
        return res
            
            
    def mixture_size_statistics(self):
        """ calculates the mean and the standard deviation of the number of
        components in mixtures """
        exp_h = np.exp(self.hs)
        denom = 1 + exp_h
        l_mean = np.sum(exp_h/denom)
        l_var = np.sum(exp_h/denom**2)
        
        return l_mean, np.sqrt(l_var)
    
    
    def set_commonness(self, scheme='const', mean_mixture_size=1, **kwargs):
        """ picks a commonness vector according to the supplied parameters """
        self.hs = np.empty(self.Ns)
        
        if scheme == 'const':
            # all substrates are equally likely
            self.hs[:] = np.log(mean_mixture_size/(self.Ns- mean_mixture_size))
        
        elif scheme == 'singular':
            # the first substrate has a different commonness than the others 
            p1 = kwargs.pop('p1', 0.5)
            assert 0 <= p1 <= 1
            self.hs[0] = np.log(p1/(1 - p1))
            self.hs[1:] = np.log((p1 - mean_mixture_size) / 
                                 (1 + mean_mixture_size - self.Ns - p1))
            
        elif scheme == 'geometric':
            # substrates have geometrically decreasing commonness 
            alpha = kwargs.pop('alpha', 0.9)
            
            if alpha == 1:
                p0 = mean_mixture_size/self.Ns
            else:
                p0 = mean_mixture_size * (1 - alpha) / (1 - alpha**self.Ns)
                
            if p0 > 1:
                raise ValueError('It is not possible to choose commonness '
                                 'parameters such that the mean mixture size '
                                 'is %g for alpha=%g'
                                 % (mean_mixture_size, alpha))
                
            i = np.arange(1, self.Ns + 1)
            self.hs = np.log(p0 * alpha**i/(alpha - p0 * alpha**i))
            
        else:
            raise ValueError('Unknown commonness scheme `%s`' % scheme)
            

        
def test_consistency():
    """ does some simple consistency tests """
    # construct random model
    Ns = np.random.randint(10, 200)
    Nr = np.random.randint(2, 6)
    hval = np.random.random() - 0.5
    frac = np.random.random()
    model = ReceptorLibraryBase(Ns, Nr, [hval]*Ns, frac)
    
    # probability of having d_s components in a mixture for h_i = h
    d_s = np.arange(0, Ns + 1)
    p_m = scipy.misc.comb(Ns, d_s) * np.exp(hval*d_s)/(1 + np.exp(hval))**Ns
    
    assert np.allclose(p_m, model.mixture_size_distribution())

    # test random commonness and the associated distribution
    model.hs = np.random.random(size=Ns)
    dist = model.mixture_size_distribution()
    assert np.allclose(dist.sum(), 1)
    ks = np.arange(0, Ns + 1)
    dist_mean = (ks*dist).sum()
    dist_var = (ks*ks*dist).sum() - dist_mean**2 
    assert np.allclose((dist_mean, np.sqrt(dist_var)),
                       model.mixture_size_statistics())
    
    # test setting the commonness
    mean_mixture_size = np.random.randint(1, 10)
    commoness_schemes = [('const', {}),
                         ('singular', {'p0': np.random.random()}),
                         ('geometric', {'alpha':0.9 + 0.1*np.random.random()})]
    
    for scheme, params in commoness_schemes:
        model.set_commonness(scheme, mean_mixture_size, **params)
        assert np.allclose(model.mixture_size_statistics()[0], mean_mixture_size)
    
    

if __name__ == '__main__':
    test_consistency()


        
    
    