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
    def get_random_arguments(cls, **kwargs):
        """ create random arguments for creating test instances """
        args = super(ReceptorLibraryTheory, cls).get_random_arguments()
        frac = kwargs.get('frac', np.random.random())
        return args + [frac]


    def activity_single(self):
        """ return the probability with which a single receptor is activated 
        by typical mixtures """
        return 1 - np.prod(1 - self.frac*self.substrate_probability)


    def mutual_information(self, with_correlations=False, approx_prob=False):
        """ return a theoretical estimate of the mutual information between
        input and output """
        if len(np.unique(self.commonness)) > 1:
            raise RuntimeError('The estimate only works for homogeneous '
                               'mixtures so far.')
        p_s = self.substrate_probability[0]
        
        # probability that a single receptor and a pair is activated 
        if approx_prob:
            # use approximate formulas for calculating the probabilities
            p1 = self.Ns * self.frac * p_s
            p2 = self.Ns * self.frac**2 * p_s
        else:
            # use better formulas for calculating the probabilities 
            p1 = 1 - (1 - self.frac * p_s)**self.Ns
            p2 = 1 - (1 - self.frac**2 * p_s)**self.Ns
        
        if p1 == 0:
            # receptors are never activated
            return 0
        elif with_correlations:
            # use the estimated formula that includes the effects of correlations
            corr1 = self.Nr*(self.Nr - 1)*(1 - 2*p1 + 0.75*p2)*p2
            corr2 = self.Nr*(self.Nr - 1)*(self.Nr - 2)*p2**2
            return self.Nr - 0.5*self.Nr*(1 - 2*p1)**2 - corr1 - corr2
        else:
            # use the simple formula where receptors are considered independent
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
            if len(np.unique(self.commonness)) > 1:
                raise RuntimeError('The estimate only works for homogeneous '
                                   'mixtures so far.')
            p0 = self.substrate_probability[0]
            
        # calculate the fraction for the homogeneous case
        return (1 - 2**(-1/self.Ns))/p0
    
    