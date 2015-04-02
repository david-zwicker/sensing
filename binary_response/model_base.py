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
            self.commonness = np.zeros(self.Ns) 
        else:
            assert len(hs) == self.Ns
            self.commonness = np.array(hs)
        self.frac = frac

    
    @property
    def commonness(self):
        """ return the commonness vector """
        return self._hs
    
    @commonness.setter
    def commonness(self, hs):
        """ sets the commonness and the associated substrate probability """
        if len(hs) != self.Ns:
            raise ValueError('Length of the commonness vector must match the '
                             'number of substrates.')
        self._hs = hs
    
    
    @property
    def substrate_probability(self):
        """ return the probability of finding each substrate """
        return 1/(1 + np.exp(-self._hs))
    
    @substrate_probability.setter
    def substrate_probability(self, ps):
        """ sets the substrate probability and the associated commonness """
        if len(ps) != self.Ns:
            raise ValueError('Length of the probability vector must match the '
                             'number of substrates.')
        if np.any(ps < 0) or np.any(ps > 1):
            raise ValueError('All probabilities must be within [0, 1]')
        
        with np.errstate(all='ignore'):
            # self._hs = np.log(ps/(1 + ps))
            self._hs = np.log(ps) - np.log1p(-ps)
            
            
    def mixture_size_distribution(self):
        """ calculates the probabilities of finding a mixture with a given
        number of components. Returns an array of length Ns + 1 of probabilities
        for finding mixtures with the number of components given by the index
        into the array """
        res = np.zeros(self.Ns + 1)
        res[0] = 1
        # iterate over each substrate and consider its individual probability
        for k, p in enumerate(self.substrate_probability, 1):
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
        prob_s = self.substrate_probability
        l_mean = np.sum(prob_s)
        l_var = np.sum(prob_s/(1 + np.exp(self._hs)))
        
        return l_mean, np.sqrt(l_var)
    
    
    def set_commonness(self, scheme='const', mean_mixture_size=1, **kwargs):
        """ picks a commonness vector according to the supplied parameters:
        `mean_mixture_size` is determines the mean number of components in each
        mixture. The value of the commonness vector are furthermore influenced  
        by the `scheme`, which can be any of the following:
            `const`: all substrates are equally probable
            `single`: the first substrate has a different probability, which can
                either be specified directly by supplying the parameter `p1` or
                the `ratio` between p1 and the probabilities of the other
                substrates can be specified.
            `geometric`: the probability of substrates decreases by a factor of
                `alpha` from each substrate to the next.        
        """
        if scheme == 'const':
            # all substrates are equally likely
            ps = np.full(self.Ns, mean_mixture_size/self.Ns)
        
        elif scheme == 'single':
            # the first substrate has a different commonness than the others
            ps = np.empty(self.Ns)
            if 'p1' in kwargs:
                # use the given probability for the first substrate
                ps[0] = kwargs['p1']
                ps[1:] = (mean_mixture_size - ps[0]) / (self.Ns - 1)
                 
            elif 'p_ratio' in kwargs:
                # use the given ratio between the first and the other substrates
                ratio = kwargs['p_ratio']
                denom = self.Ns + ratio - 1
                ps[0] = mean_mixture_size * ratio / denom
                ps[1:] = mean_mixture_size / denom
                
            else:
                raise ValueError('Either `p1` or `p_ratio` must be given')

            
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
                
            ps = p0 * alpha**np.arange(self.Ns)
            
        else:
            raise ValueError('Unknown commonness scheme `%s`' % scheme)
        
        # set the probability which also calculates the commonness
        self.substrate_probability = ps


        
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
    hs = np.random.random(size=Ns)
    model.commonness = hs
    assert np.allclose(hs, model.commonness)
    model.substrate_probability = model.substrate_probability
    assert np.allclose(hs, model.commonness)
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
                         ('single', {'p1': np.random.random()}),
                         ('single', {'p_ratio': 0.1 + np.random.random()}),
                         ('geometric', {'alpha':0.9 + 0.1*np.random.random()})]
    
    for scheme, params in commoness_schemes:
        model.set_commonness(scheme, mean_mixture_size, **params)
        assert np.allclose(model.mixture_size_statistics()[0],
                           mean_mixture_size)
    
    

if __name__ == '__main__':
    test_consistency()


        
    
    