'''
Created on Mar 31, 2015

@author: David Zwicker <dzwicker@seas.harvard.edu>
'''

from __future__ import division

import numpy as np
import scipy.misc

from .library_base import LibraryBinaryBase



LN2 = np.log(2)



def binom(N, p):
    """ calculate the probability mass function for the binomial distribution
    of `N` experiments with individual probability `p` """
    k = np.arange(0, N + 1)
    return scipy.misc.comb(N, k) * p**k * (1 - p)**(N - k)



class LibraryBinaryUniform(LibraryBinaryBase):
    """ represents a single receptor library with random entries. The only
    parameters that characterizes this library is the density of entries. """


    def __init__(self, num_substrates, num_receptors, density=1,
                 parameters=None):
        """ initialize the receptor library by setting the number of receptors,
        the number of substrates it can respond to, and the fraction `density`
        of substrates a single receptor responds to """
        super(LibraryBinaryUniform, self).__init__(num_substrates,
                                                   num_receptors, parameters)
        self.density = density


    @property
    def init_arguments(self):
        """ return the parameters of the model that can be used to reconstruct
        it by calling the __init__ method with these arguments """
        args = super(LibraryBinaryUniform, self).init_arguments
        args['density'] = self.density
        return args


    @classmethod
    def get_random_arguments(cls, **kwargs):
        """ create random arguments for creating test instances """
        args = super(LibraryBinaryUniform, cls).get_random_arguments(**kwargs)
        args['density'] = kwargs.get('density', np.random.random())
        return args


    def activity_single(self):
        """ return the probability with which a single receptor is activated 
        by typical mixtures """
        if self.has_correlations:
            raise NotImplementedError('Not implemented for correlated mixtures')

        return 1 - np.prod(1 - self.density * self.substrate_probabilities)


    def mutual_information(self, approx_prob=False, use_polynom=False, 
                           with_crosstalk=False):
        """ return a theoretical estimate of the mutual information between
        input and output.
            `approx_prob` determines whether a linear approximation should be
                used to calculate the probabilities that receptors are active
            `use_polynom` determines whether a polynomial approximation for the
                mutual information should be used
            `with_crosstalk` determines whether the crosstalk between receptors
                should also be included. Note that the cross talk will be
                approximated by a polynomial expression independent of the
                `use_polynom` argument.
        """
        if self.has_correlations:
            raise NotImplementedError('Not implemented for correlated mixtures')

        p_i = self.substrate_probabilities
        
        # get probability q_n and q_nm that receptors are activated 
        if approx_prob:
            # use approximate formulas for calculating the probabilities
            q_n = self.density * p_i.sum()
            q_nm = self.density**2 * p_i.sum()
        else:
            # use better formulas for calculating the probabilities 
            q_n = 1 - np.prod(1 - self.density * p_i)
            q_nm = 1 - np.prod(1 - self.density**2 * p_i)
        
        # calculate the mutual information with requested method
        if use_polynom:
            # calculate MI using Taylor approximation
            MI = self.Nr - 0.5/LN2 * self.Nr * (1 - 2*q_n)**2

        elif q_n == 0 or q_n == 1:
            # receptors are never or always activated
            MI = 0
            
        else:
            # calculate MI by assuming that receptors are independent

            # calculate the information a single receptor contributes            
            H_r = -(q_n*np.log2(q_n) + (1 - q_n)*np.log2(1 - q_n))
            
            # calculate the MI assuming that receptors are independent
            # This expression assumes that each receptor provides a fractional 
            # information H_r/N_s. Some of the information will be overlapping
            # and the resulting MI is thus smaller than the naive estimate:
            #     MI < N_r * H_r
            MI = self.Ns - self.Ns*(1 - H_r/self.Ns)**self.Nr
           
        if with_crosstalk:
            Nr = self.Nr
            MI -= 1/LN2 * (Nr**2 - Nr) * (0.75*q_nm + 2*q_n - 1) * q_nm
            MI -= 0.5/LN2 * (Nr**3 - 3*Nr**2 + 2*Nr) * q_nm**2
            
        # determine the entropy of the mixtures
        H_m = -np.sum(p_i*np.log2(p_i) + (1 - p_i)*np.log2(1 - p_i))
        # limit the MI to the mixture entropy
        return min(MI, H_m)
        
        
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
        if not assume_homogeneous and len(np.unique(self.commonness)) > 1:
            # mixture is heterogeneous
            raise RuntimeError('The estimate only works for homogeneous '
                               'mixtures so far.')
                
        # mean probability of finding a specific substrate in a mixture
        p0 = self.substrate_probabilities.mean()
            
        # calculate the fraction for the homogeneous case
        return (1 - 2**(-1/self.Ns))/p0
    
    
    def get_optimal_library(self):
        """ returns an estimate for the optimal parameters for the random
        interaction matrices """
        return {'density': self.density_optimal(assume_homogeneous=True)}
        
        