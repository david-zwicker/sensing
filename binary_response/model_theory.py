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



class ReceptorLibraryUniform(ReceptorLibraryBase):
    """ represents a single receptor library with random entries. The only
    parameters that characterizes this library is the density of entries. """


    def __init__(self, num_substrates, num_receptors, hs=None, density=1):
        """ initialize the receptor library by setting the number of receptors,
        the number of substrates it can respond to, the weights `hs` of the 
        substrates, and the fraction `density` of substrates a single receptor
        responds to """
        super(ReceptorLibraryUniform, self).__init__(num_substrates,
                                                     num_receptors, hs)
        self.density = density


    @property
    def init_arguments(self):
        """ return the parameters of the model that can be used to reconstruct
        it by calling the __init__ method with these arguments """
        args = super(ReceptorLibraryUniform, self).init_arguments
        args['density'] = self.density
        return args


    @classmethod
    def get_random_arguments(cls, **kwargs):
        """ create random arguments for creating test instances """
        args = super(ReceptorLibraryUniform, cls).get_random_arguments()
        density = kwargs.get('density', np.random.random())
        return args + [density]


    def activity_single(self):
        """ return the probability with which a single receptor is activated 
        by typical mixtures """
        return 1 - np.prod(1 - self.density * self.substrate_probability)


    def mutual_information(self, with_correlations=True, approx_prob=False):
        """ return a theoretical estimate of the mutual information between
        input and output """
        assert len(np.unique(self.commonness)) == 1
        p_s = self.substrate_probability
        
        # probability that a single receptor and a pair is activated 
        if approx_prob:
            # use approximate formulas for calculating the probabilities
            p1 = self.Ns * self.density * p_s
            p2 = self.Ns * self.density**2 * p_s
        else:
            # use better formulas for calculating the probabilities 
            p1 = self.activity_single()
            p2 = 1 - (1 - self.density**2 * p_s)**self.Ns
        
        if p1 == 0:
            # receptors are never activated
            return 0
        elif with_correlations:
            # use the estimated formula that includes the effects of correlations
            corr = self.Nr*(self.Nr - 1)*(1 - 2*p1 + 0.75*p2)*p2
            return self.Nr - 0.5*self.Nr*(1 - 2*p1)**2 - corr
        else:
            # use the simple formula where receptors are considered independent
            return -self.Nr*(p1*np.log2(p1) + (1 - p1)*np.log2(1 - p1))
        
        
    def density_optimal(self, assume_homogeneous=False):
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
            assert len(np.unique(self.commonness)) == 1
            p0 = self.substrate_probability[0]
            
        # calculate the fraction for the homogeneous case
        Ns2 = 2**(1/self.Ns)
        return (Ns2 - 1)/(Ns2 * p0)
    
    