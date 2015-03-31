'''
Created on Mar 31, 2015

@author: David Zwicker <dzwicker@seas.harvard.edu>
'''

from __future__ import division

import numpy as np
import scipy.misc



class ReceptorLibraryTheory(object):
    """ represents a single receptor library """

    def __init__(self, num_receptors, num_substrates, hs, frac=1):
        """ initialize the receptor library by setting the number of receptors,
        the number of substrates it can respond to, the weights `hs` of the 
        substrates, and the fraction `frac` of substrates a single receptor
        responds to """
        assert len(hs) == num_substrates
        
        self.Nr = num_receptors
        self.Ns = num_substrates
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
    
    
    def prob_activity(self):
        """ return the probability with which a given receptor is activated """
        assert len(np.unique(self.hs)) == 1
        h = self.hs[0]
        
        if self.frac == 0:
            return 0
        d = np.arange(0, self.Ns + 1)
        terms = (scipy.misc.comb(self.Ns, d) * 
                 np.exp(h*d)/(1 + np.exp(h))**self.Ns *
                 (1 - (1 - self.frac*d/self.Ns)**self.Ns))
        return terms.sum()


    def mutual_information(self):
        """ return theoretical estimate of the mutual information between input
        and output """
        assert len(np.unique(self.hs)) == 1
        Pa_val = self.prob_activity()
        if Pa_val == 0:
            return 0
        else:
            return -self.Nr*Pa_val*np.log(Pa_val)

