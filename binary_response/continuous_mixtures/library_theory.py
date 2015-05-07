'''
Created on Mar 31, 2015

@author: David Zwicker <dzwicker@seas.harvard.edu>
'''

from __future__ import division

import warnings

import numpy as np
from scipy import stats, integrate, optimize, special

from utils.math_distributions import (DeterministicDistribution,
                                      HypoExponentialDistribution) 
from .library_base import LibraryContinuousBase



class LibraryContinuousLogNormal(LibraryContinuousBase):
    """ represents a single receptor library with random entries. The only
    parameters that characterizes this library is the density of entries. """


    def __init__(self, num_substrates, num_receptors, mean_sensitivity=1,
                 sigma=0.1, parameters=None):
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
        args['sigma'] = kwargs.get('sigma', 0.1*np.random.random())
        return args


    @classmethod
    def from_numeric(cls, numeric_model, mean_sensitivity=None, sigma=None,
                     parameters=None):
        """ creates an instance of this class by using parameters from a related
        numeric instance """
        # set parameters
        kwargs = {'parameters': parameters}
        if mean_sensitivity is not None:
            kwargs['mean_sensitivity'] = mean_sensitivity
        elif numeric_model.int_mat is not None:
            kwargs['mean_sensitivity'] = numeric_model.int_mat.mean()
        if sigma is not None:
            kwargs['sigma'] = sigma
        
        # create the object
        obj = cls(numeric_model.Ns, numeric_model.Nr, **kwargs)
        
        # copy the commonness from the numeric model
        obj.commonness = numeric_model.commonness
        
        return obj


    def activity_single(self):
        """ return the probability with which a single receptor is activated 
        by typical mixtures """
        hs = self.commonness
        
        if self.sigma == 0:
            # simple case in which the interaction matrix elements are the same:
            #     I_ai = self.mean_sensitivity
            # this is the limiting case
            if self.is_homogeneous:
                # evaluate the full integral for the case where all substrates
                # are equally likely
                dist = stats.gamma(a=self.Ns, scale=-1/hs[0])
                prob_a0 = dist.cdf(1/self.mean_sensitivity) - dist.cdf(0)
                
            else:
                # the probability of the total concentration c_tot is given
                # by a hypoexponential function with the following cdf:
                warnings.warn('The numerical implementation of the cdf of the '
                              'hypoexponential function is very unstable and '
                              'the results cannot be trusted.')
                c_means = self.get_concentration_means()
                cdf_ctot = HypoExponentialDistribution(c_means).cdf
                prob_a0 = cdf_ctot(1/self.mean_sensitivity)

        else:
            # finite-width distribution of interaction matrix elements
            if self.is_homogeneous:
                # FIXME: this is the result for the simple case where all
                # I_ai are equal for a given a
                dist = stats.gamma(a=self.Ns, scale=-1/hs[0])
                cdf = self.int_mat_distribution.cdf
                integrand = lambda c: cdf(1/c) * dist.pdf(c)
                prob_a0 = integrate.quad(integrand, 0, np.inf)[0]
                
            else:
                # finite-width distribution with heterogeneous mixtures
                raise NotImplementedError
#                 prob_a0 = 1
#                 for h in hs:
#                     integrand = lambda c: np.exp(h*c) * cdf(1/c)
#                     prob_a0 *= -h * integrate.quad(integrand, 0, np.inf)[0]
                
        return 1 - prob_a0
    
    
    def activity_single_gaussian(self):
        """ return the probability with which a single receptor is activated 
        by typical mixtures using an gaussian approximation """
        hs = self.commonness

        if self.sigma == 0:
            # simple case in which the interaction matrix elements are the same:
            #     I_ai = self.mean_sensitivity
            # this is the limiting case

            # replace the hypoexponential distribution by a normal one
            mean = -np.sum(1/hs) 
            denom = np.sqrt(2) * np.sqrt(np.sum(1/hs**2)) # sqrt(2) * std
            c_max = 1/self.mean_sensitivity
            prob_a0 = 0.5*(special.erf((c_max - mean)/denom)
                           + special.erf(mean/denom))
            prob_a1 = 1 - prob_a0
            
        else:
            # finite-width distribution of interaction matrix elements
            # => replace the complicated distributions by normal distributions
            # and estimate the activity with this
            I0 = self.mean_sensitivity
            sigma2 = self.sigma**2
            
            # determine the parameters describing the distribution of the
            # z-values as given by z = I_ai * c_i drawn from their
            # respective distributions
            mean_z = -I0/hs * np.exp(0.5*sigma2)
            var_z = (I0/hs)**2 * (2*np.exp(sigma2) - 1) * np.exp(sigma2)
            
            # use these values to calculate the probability that the sum
            # of the z-values is larger than the threshold 1 assuming a
            # normal distribution under the hood
            arg = (mean_z.sum() - 1)/np.sqrt(2 * var_z.sum())
            prob_a1 = 0.5*(1 + special.erf(arg))
            
        return prob_a1


    def get_optimal_mean_sensitivity(self, estimate=None, gaussian=False):
        """ estimates the optimal average value of the interaction matrix
        elements """  
        if estimate is None:
            c_mean = self.get_concentration_means().mean()
            if self.sigma < 1e-8:
                # simple estimate for homogeneous mixtures with sigma=0
                term1 = self.Ns * c_mean
                term2 = (np.sqrt(2*self.Ns) * c_mean
                         * special.erfinv(1 - special.erf(np.sqrt(0.5*self.Ns))))
                estimate = 1/(term1 + term2)
            else:
                estimate = np.exp(-0.5*self.sigma**2)/(self.Ns * c_mean)
        
        return estimate
        
        # create a copy of the current object for optimization
        obj = self.copy()
        if gaussian:
            def opt_goal(I0):
                """ helper function to find optimum numerically """ 
                obj.mean_sensitivity = I0
                return 0.5 - obj.activity_single_gaussian()
        else:
            def opt_goal(I0):
                """ helper function to find optimum numerically """ 
                obj.mean_sensitivity = I0
                return 0.5 - obj.activity_single()
        
        try:
            result = optimize.newton(opt_goal, estimate)
        except RuntimeError:
            result = np.nan
            
        return result 
            

        