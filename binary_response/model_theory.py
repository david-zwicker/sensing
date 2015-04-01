'''
Created on Mar 31, 2015

@author: David Zwicker <dzwicker@seas.harvard.edu>
'''

from __future__ import division

import numpy as np
import scipy.misc

from .model_base import ReceptorLibraryBase



def binom(N, p):
    k = np.arange(0, N + 1)
    return scipy.misc.comb(N, k) * p**k * (1 - p)**(N - k)



class ReceptorLibraryTheory(ReceptorLibraryBase):
    """ represents a single receptor library """
    
    
    def activity_single(self):
        """ return the probability with which a single receptor is activated 
        by typical mixtures """
        assert len(np.unique(self.hs)) == 1
        h = self.hs[0]
        
        if self.frac == 0:
            return 0

#         # consider all possible number of components in mixtures
#         d_s = np.arange(0, self.Ns + 1)
#         
#         # probability that a receptor is activated by d_s substrates
#         p_a1 = 1 - (1 - self.frac)**d_s
#         
#         # probability P(|m| = d_s) of having d_s components in a mixture
#         p_m = scipy.misc.comb(self.Ns, d_s) * np.exp(h*d_s)/(1 + np.exp(h))**self.Ns
# 
#         # probability that a receptor is activated by a typical mixture
#         val_ex = np.sum(p_m * p_a1)
        
        term = (1 + (1 - self.frac)*np.exp(h))/(1 + np.exp(h))
        val = 1 - term ** self.Ns
        
        return val


    def mutual_information(self):
        """ return a theoretical estimate of the mutual information between
        input and output """
        assert len(np.unique(self.hs)) == 1
        
        # probability that a single receptor is activated
        p1 = self.activity_single()
        
        if p1 == 0:
            return 0
        else:
            
#             # output patterns consist of Nr receptors that are all activated
#             # with probability p1
#             
#             # probability of finding a particular pattern of exactly d_r
#             # activated receptors
#             d_r = np.arange(0, self.Nr + 1)
#             p_a = p1**d_r * (1 - p1)**(self.Nr - d_r)
#             
#             # number of possibilities of activity patterns with exactly d_r
#             # activated receptors
#             binom = scipy.misc.comb(self.Nr, d_r)
#             
#             # mutual information from the probabilities and the frequency of 
#             # finding these patterns
#             val1 = -sum(binom * p_a * np.log2(p_a))
            
            val2 = -self.Nr*(p1*np.log2(p1) + (1 - p1)*np.log2(1 - p1))
            
            return val2
        
        
    def frac_optimal(self):
        """ return the estimated optimal activity fraction for the simple case
        where all h are the same. The estimate relies on an approximation that
        all receptors are independent and is thus independent of the number of 
        receptors. The estimate is thus only good in the limit of low Nr """
        assert len(np.unique(self.hs)) == 1
        
        Ns2 = 2**(1/self.Ns)
        eh = np.exp(self.hs[0])
        return (Ns2 - 1)/Ns2 * (1 + eh)/eh



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
