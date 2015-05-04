'''
Created on Mar 31, 2015

@author: David Zwicker <dzwicker@seas.harvard.edu>
'''

from __future__ import division

import numpy as np
from scipy import stats, integrate

from utils.math_distributions import DeterministicDistribution
from .library_base import LibraryContinuousBase



class LibraryContinuousLogNormal(LibraryContinuousBase):
    """ represents a single receptor library with random entries. The only
    parameters that characterizes this library is the density of entries. """


    def __init__(self, num_substrates, num_receptors, mean_sensitivity=1,
                 sigma=1, parameters=None):
        """ initialize the receptor library by setting the number of receptors,
        the number of substrates it can respond to, the weights `hs` of the 
        substrates, and the fraction `density` of substrates a single receptor
        responds to """
        super(LibraryContinuousLogNormal, self).__init__(num_substrates,
                                                         num_receptors,
                                                         parameters)
        self.mean_sensitivity = mean_sensitivity
        self.sigma = sigma
        
        
    @property
    def int_mat_distribution(self):
        """ returns the probability distribution for the interaction matrix """
        if self.sigma == 0:
            return DeterministicDistribution(loc=self.mean_sensitivity)
        else:
            return stats.lognorm(scale=self.mean_sensitivity, s=self.sigma)


    @property
    def init_arguments(self):
        """ return the parameters of the model that can be used to reconstruct
        it by calling the __init__ method with these arguments """
        args = super(LibraryContinuousLogNormal, self).init_arguments
        args['mean_sensitivity'] = self.mean_sensitivity
        args['sigma'] = self.sigma
        return args


    @classmethod
    def get_random_arguments(cls, **kwargs):
        """ create random arguments for creating test instances """
        args = super(LibraryContinuousLogNormal, cls).get_random_arguments(**kwargs)
        I0 = np.random.random() + 0.5
        args['mean_sensitivity'] = kwargs.get('mean_sensitivity', I0)
        args['sigma'] = kwargs.get('sigma', np.random.random())
        return args


    def activity_single(self):
        """ return the probability with which a single receptor is activated 
        by typical mixtures """
        hs = self.commonness
        
        if self.sigma == 0:
            # special case in which the interaction matrix elements are the same
            prob_a = np.prod(1 - np.exp(hs/self.mean_sensitivity))
            
        else:
            # finite-width distribution of interaction matrix elements
            cdf = self.int_mat_distribution.cdf
            if self.is_homogeneous:
                # finite-width distribution, but homogeneous mixtures
                h = hs[0]
                integrand = lambda c: -h*np.exp(h*c) * cdf(1/c)
                prob_a = integrate.quad(integrand, 0, np.inf)[0]
                prob_a **= self.Ns
                
            else:
                # finite-width distribution with heterogeneous mixtures
                prob_a = 1
                for h in hs:
                    integrand = lambda c: -h*np.exp(h*c) * cdf(1/c)
                    prob_a *= integrate.quad(integrand, 0, np.inf)[0]
                
        return 1 - prob_a



        