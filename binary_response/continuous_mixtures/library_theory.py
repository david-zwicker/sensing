'''
Created on Mar 31, 2015

@author: David Zwicker <dzwicker@seas.harvard.edu>
'''

from __future__ import division

import numpy as np
from scipy import stats, integrate

from .library_base import LibraryContinuousBase



class LibraryContinuousLogNormal(LibraryContinuousBase):
    """ represents a single receptor library with random entries. The only
    parameters that characterizes this library is the density of entries. """


    def __init__(self, num_substrates, num_receptors, parameters=None,
                 mean_sensitivity=1, sigma=1):
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
        args = super(LibraryContinuousLogNormal, cls).get_random_arguments()
        mean_sensitivity = kwargs.get('mean_sensitivity', np.random.random())
        sigma = kwargs.get('sigma', np.random.random())
        return args + [mean_sensitivity, sigma]


    def activity_single(self):
        """ return the probability with which a single receptor is activated 
        by typical mixtures """
        hs = self.commonness
        cdf = self.int_mat_distribution.cdf
        
        if self.is_homogeneous:
            h = hs[0]
            integrand = lambda c: -h*np.exp(h*c) * cdf(1/c)
            prob_a = integrate.quad(integrand, 0, np.inf)[0]
            prob_a **= self.Ns
            
        else:
            prob_a = 1
            for h in hs:
                integrand = lambda c: -h*np.exp(h*c) * cdf(1/c)
                prob_a *= integrate.quad(integrand, 0, np.inf)[0]
                
        return 1 - prob_a



        