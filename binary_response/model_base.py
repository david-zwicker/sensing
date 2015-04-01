'''
Created on Apr 1, 2015

@author: David Zwicker <dzwicker@seas.harvard.edu>
'''

from __future__ import division

import numpy as np



class ReceptorLibraryBase(object):
    """ represents a single receptor library. This is a base class that provides
    general functionality and parameter management """


    def __init__(self, num_substrates, num_receptors, hs, frac=1):
        """ initialize the receptor library by setting the number of receptors,
        the number of substrates it can respond to, the weights `hs` of the 
        substrates, and the fraction `frac` of substrates a single receptor
        responds to """
        assert len(hs) == num_substrates
        
        self.Ns = num_substrates
        self.Nr = num_receptors
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
            r = res[:k].copy()
            res[:k] *= 1 - p  #< substrate not in the mixture 
            res[1:k+1] += r*p #< substrate in the mixture
            
        return res
            
            
    def mixture_size_statistics(self):
        """ calculates the mean and the standard deviation of the number of
        components in mixtures """
        exp_h = np.exp(self.hs)
        denom = 1 + exp_h
        l_mean = np.sum(exp_h/denom)
        l_var = np.sum(exp_h/denom**2)
        
        return l_mean, np.sqrt(l_var)
    
    
    def set_commonness_geometric(self, mean_mixture_size=1, alpha=0.9):
        """ picks a commonness vector according to the supplied parameters """
        p0 = mean_mixture_size * (1 - alpha) / (1 - alpha**self.Ns) 
        i = np.arange(1, self.Ns + 1)
        self.hs = np.log(p0 * alpha**i/(alpha - p0 * alpha**i))
        
        
        
def test_consistency():
    """ does some simple consistency tests """
    # construct random model
    Ns = np.random.randint(100, 200)
    Nr = np.random.randint(2, 6)
    hval = np.random.random() - 0.5
    frac = np.random.random()
    model = ReceptorLibraryBase(Ns, Nr, [hval]*Ns, frac)
    
    # set the commonness
    mean_mixture_size = np.random.randint(1, 10)
    alpha = 0.9 + 0.1*np.random.random()
    model.set_commonness_geometric(mean_mixture_size, alpha)

    assert np.allclose(model.mixture_size_statistics()[0], mean_mixture_size)
    
    

if __name__ == '__main__':
    test_consistency()


        
    
    