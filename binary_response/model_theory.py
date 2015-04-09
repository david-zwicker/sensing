'''
Created on Mar 31, 2015

@author: David Zwicker <dzwicker@seas.harvard.edu>
'''

from __future__ import division

import numpy as np
import scipy.misc

from .model_base import ReceptorLibraryBase



def binom(N, p):
    """ calculate the probability mass function for the binomial distribution
    of `N` experiments with individual probability `p` """
    k = np.arange(0, N + 1)
    return scipy.misc.comb(N, k) * p**k * (1 - p)**(N - k)



class ReceptorLibraryTheory(ReceptorLibraryBase):
    """ represents a single receptor library """


    def __init__(self, num_substrates, num_receptors, hs=None, frac=1):
        """ initialize the receptor library by setting the number of receptors,
        the number of substrates it can respond to, the weights `hs` of the 
        substrates, and the fraction `frac` of substrates a single receptor
        responds to """
        super(ReceptorLibraryTheory, self).__init__(num_substrates,
                                                    num_receptors, hs)
        self.frac = frac


    @property
    def init_arguments(self):
        """ return the parameters of the model that can be used to reconstruct
        it by calling the __init__ method with these arguments """
        args = super(ReceptorLibraryTheory, self).init_arguments
        args['frac'] = self.frac
        return args


    @classmethod
    def get_random_arguments(cls):
        """ create random arguments for creating test instances """
        args = super(ReceptorLibraryTheory, cls).get_random_arguments()
        frac = np.random.random()
        return args + [frac]


    def activity_single(self):
        """ return the probability with which a single receptor is activated 
        by typical mixtures """
        assert len(np.unique(self._hs)) == 1
        h = self._hs[0]
        
        if self.frac == 0:
            return 0

        term = (1 + (1 - self.frac)*np.exp(h))/(1 + np.exp(h))
        val = 1 - term ** self.Ns
        
        return val


    def mutual_information(self):
        """ return a theoretical estimate of the mutual information between
        input and output """
        assert len(np.unique(self._hs)) == 1
        
        # probability that a single receptor is activated
        p1 = self.activity_single()
        
        if p1 == 0:
            return 0
        else:
            return -self.Nr*(p1*np.log2(p1) + (1 - p1)*np.log2(1 - p1))
        
        
    def frac_optimal(self, assume_homogeneous=False):
        """ return the estimated optimal activity fraction for the simple case
        where all h are the same. The estimate relies on an approximation that
        all receptors are independent and is thus independent of the number of 
        receptors. The estimate is thus only good in the limit of low Nr.
        
        If `assume_homogeneous` is True, the calculation is also done in the
            case of heterogeneous mixtures, where the probability of the
            homogeneous system with the same average number of substrates is
            used instead.
        """
        if assume_homogeneous:
            # calculate the idealized substrate probability
            m_mean = self.mixture_size_statistics()[0]
            p0 = m_mean / self.Ns
             
        else:
            # check whether the mixtures are all homogeneous
            assert len(np.unique(self._hs)) == 1
            p0 = self.substrate_probability[0]
            
        # calculate the fraction for the homogeneous case
        Ns2 = 2**(1/self.Ns)
        return (Ns2 - 1)/(Ns2 * p0)
    
    