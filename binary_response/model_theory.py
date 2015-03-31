'''
Created on Mar 31, 2015

@author: David Zwicker <dzwicker@seas.harvard.edu>
'''

from __future__ import division

import numpy as np
import scipy.misc


def binom(N, p):
    k = np.arange(0, N + 1)
    return scipy.misc.comb(N, k) * p**k * (1 - p)**(N - k)



class ReceptorLibraryTheory(object):
    """ represents a single receptor library """

    def __init__(self, num_substrates, num_receptors, hs, frac=1):
        """ initialize the receptor library by setting the number of receptors,
        the number of substrates it can respond to, the weights `hs` of the 
        substrates, and the fraction `frac` of substrates a single receptor
        responds to """
        assert len(hs) == num_substrates
        
        self.Ns = num_substrates
        self.Nr = num_receptors
        self.hs = hs
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
    
    
    def activity_single(self):
        """ return the probability with which a single receptor is activated 
        by typical mixtures """
        assert len(np.unique(self.hs)) == 1
        h = self.hs[0]
        
        if self.frac == 0:
            return 0

        # consider all possible number of components in mixtures
        d_s = np.arange(0, self.Ns + 1)
        
        # probability that a receptor is activated by d_s substrates
        p_a1 = 1 - (1 - self.frac)**d_s
        
        # probability P(|m| = d_s) of having d_s components in a mixture
        p_m = scipy.misc.comb(self.Ns, d_s) * np.exp(h*d_s)/(1 + np.exp(h))**self.Ns

        # probability that a receptor is activated by a typical mixture
        return np.sum(p_m * p_a1)


    def mutual_information(self):
        """ return a theoretical estimate of the mutual information between
        input and output """
        assert len(np.unique(self.hs)) == 1
        
        # probability that a single receptor is activated
        p1 = self.activity_single()
        
        if p1 == 0:
            return 0
        else:
            
            # output patterns consist of Nr receptors that are all activated
            # with probability p1
            
            # probability of finding a particular pattern of exactly d_r
            # activated receptors
            d_r = np.arange(0, self.Nr + 1)
            p_a = p1**d_r * (1 - p1)**(self.Nr - d_r)
            
            # number of possibilities of activity patterns with exactly d_r
            # activated receptors
            binom = scipy.misc.comb(self.Nr, d_r)
            
            # mutual information from the probabilities and the frequency of 
            # finding these patterns
            return -sum(binom * p_a * np.log(p_a))



def test_consistency():
    """ does some simple consistency tests """
    # construct random model
    Ns = np.random.randint(10, 20)
    Nr = np.random.randint(2, 6)
    hval = np.random.random() - 0.5
    frac = np.random.random()
    model = ReceptorLibraryTheory(Ns, Nr, [hval]*Ns, frac)
    
    # probability of having d_s components in a mixture
    d_s = np.arange(0, Ns + 1)
    p_m = scipy.misc.comb(Ns, d_s) * np.exp(hval*d_s)/(1 + np.exp(hval))**Ns
    
    assert np.allclose(p_m, model.mixture_size_distribution())



if __name__ == '__main__':
    test_consistency()
