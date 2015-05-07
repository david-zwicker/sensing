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
                prob_a0 = dist.cdf(1/self.mean_sensitivity)
                
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
    
    
    def activity_single_estimate(self, method='normal'):
        """ return the probability with which a single receptor is activated 
        by typical mixtures using an gaussian approximation """
        hs = self.commonness

        if method == 'normal':
            # use a normal distribution for approximations

            if self.sigma == 0:
                # simple case in which all interaction matrix elements are the same:
                #     I_ai = self.mean_sensitivity
                # this is the limiting case
    
                # get the moments of the hypoexponential distribution
                ctot_mean = -np.sum(1/hs)
                ctot_var = np.sum(1/hs**2)
                # these values directly parameterize the normal distribution

                # evaluate the fraction of values that exceeds the threshold
                # value given by c_min = 1/self.mean_sensitivity. We thus
                # evaluate the integral from c_min to infinity, which equals
                #     1/2 Erfc[(cmin - ctot_mean)/Sqrt[2 * ctot_var]]
                # according to Mathematica
                c_min = 1/self.mean_sensitivity
                arg = (c_min - ctot_mean) / np.sqrt(2*ctot_var)
                prob_a1 = 0.5 * special.erfc(arg)
                
            else:
                # more complicated case where the distribution of interaction
                # matrix elements has a finite width
                I0 = self.mean_sensitivity
                sigma2 = self.sigma**2
                
                # we first determine the mean and the variance of the
                # distribution of z = I_ai * c_i, which is the distribution for
                # a single matrix element I_ai multiplied the concentration of a
                # single substrate
                zi_mean = -I0/hs * np.exp(0.5*sigma2)
                zi_var = (I0/hs)**2 * (2*np.exp(sigma2) - 1) * np.exp(sigma2)
                # these values directly parameterize the normal distribution

                # add up all the N_s distributions to find the probability
                # distribution for determining the activity of a receptor.
                # Since, these are normal distributions, both the means and the
                # variances just add up
                z_mean = zi_mean.sum()
                z_var = zi_var.sum()
                
                # integrate the resulting normal distribution from 1 to infinity
                # to determine the probability of exceeding 1
                # Mathematica says that this integral equals
                #     1/2 Erfc[(1 - z_mean)/Sqrt[2 * z_var]]
                prob_a1 = 0.5 * special.erfc((1 - z_mean) / np.sqrt(2*z_var))
        
        elif method == 'gamma':
            # use a gamma distribution for approximations
             
            if self.sigma == 0:
                # simple case in which the interaction matrix elements are the same:
                #     I_ai = self.mean_sensitivity
                # this is the limiting case
    
                # get the moments of the hypoexponential distribution
                ctot_mean = -np.sum(1/hs)
                ctot_var = np.sum(1/hs**2)
                
                # calculate the parameters of the associated gamma distribution
                alpha = ctot_mean**2 / ctot_var
                beta = ctot_var / ctot_mean
                
                # evaluate the fraction of values that exceeds the threshold
                # value given by c_min = 1/self.mean_sensitivity. We thus
                # evaluate the integral from c_min to infinity, which equals
                #     Gamma[\[Alpha], cMin/\[Beta]]/Gamma[\[Alpha]]
                # according to Mathematica
                c_min = 1/self.mean_sensitivity
                prob_a1 = special.gammaincc(alpha, c_min/beta)
            
            else:
                # more complicated case where the distribution of interaction
                # matrix elements has a finite width
                I0 = self.mean_sensitivity
                sigma2 = self.sigma**2
                
                # we first determine the mean and the variance of the
                # distribution of z = I_ai * c_i, which is the distribution for
                # a single matrix element I_ai multiplied the concentration of a
                # single substrate
                h_mean = hs.mean()
                z_mean = -I0/h_mean * np.exp(0.5*sigma2)
                z_var = (I0/h_mean)**2 * (2*np.exp(sigma2) - 1) * np.exp(sigma2)
                
                # calculate the parameters of the associated gamma distribution
                alpha = z_mean**2 / z_var
                beta = z_var / z_mean
                
                # add up all the N_s distributions to find the probability
                # distribution for determining the activity of a receptor
                alpha *= self.Ns
                # this assumes that beta is the same for all individual
                # substrates, which is only the case for homogeneous mixtures
                if not self.is_homogeneous:
                    warnings.warn('The estimate using gamma distributions '
                                  'currently assumes that all substrates have '
                                  'the same distribution.')
                
                # integrate the gamma distribution from 1 to infinity to
                # determine the probability of exceeding 1
                # Mathematica says that this integral equals
                #     Gamma[\[Alpha], 1/\[Beta]]/Gamma[\[Alpha]]
                prob_a1 = special.gammaincc(alpha, 1/beta)
                
        else:
            raise ValueError('Unknown estimation method `%s`' % method)
                
        return prob_a1


    def get_optimal_mean_sensitivity(self, estimate=None, approximation=None):
        """ estimates the optimal average value of the interaction matrix
        elements """  
        if estimate is None:
            c_mean = self.get_concentration_means().mean()
            if self.sigma < 1e-8:
                # simple estimate for homogeneous mixtures with sigma=0
                term1 = self.Ns * c_mean
                term2 = (np.sqrt(2*self.Ns) * c_mean
                         * special.erfinv(1 - special.erf(np.sqrt(self.Ns/2))))
                estimate = 1/(term1 + term2)
            else:
                estimate = np.exp(-0.5 * self.sigma**2)/(self.Ns * c_mean)
        
        # find best mean_sensitivity by optimizing until the average receptor
        # activity is 0.5
        obj = self.copy() #< copy of the current object for optimization
        
        # check which approximation to use
        if approximation is None or approximation == 'none':
            def opt_goal(I0):
                """ helper function to find optimum numerically """ 
                obj.mean_sensitivity = I0
                return 0.5 - obj.activity_single()
            
        else:
            def opt_goal(I0):
                """ helper function to find optimum numerically """ 
                obj.mean_sensitivity = I0
                return 0.5 - obj.activity_single_estimate(approximation)
        
        try:
            result = optimize.newton(opt_goal, estimate)
        except RuntimeError:
            result = np.nan
            
        return result 
            
            
    def mutual_information(self, approximation=None):
        """ return a theoretical estimate of the mutual information between
        input and output """
        if approximation is None or approximation == 'none':
            p_r = self.activity_single()
        else:
            p_r = self.activity_single_estimate(approximation)
            
        if p_r == 0 or p_r == 1:
            # receptors are never or always activated
            return 0

        else:
            # calculate the information a single receptor contributes            
            H_r = -(p_r*np.log2(p_r) + (1 - p_r)*np.log2(1 - p_r))
            # calculate the MI assuming that receptors are independent
            MI = self.Ns - self.Ns*(1 - H_r/self.Ns)**self.Nr
            return MI
        
        