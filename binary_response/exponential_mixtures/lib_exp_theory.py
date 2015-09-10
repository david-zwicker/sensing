'''
Created on Mar 31, 2015

@author: David Zwicker <dzwicker@seas.harvard.edu>
'''

from __future__ import division

import warnings

import numpy as np
from scipy import stats, integrate, optimize, special

from .lib_exp_base import LibraryExponentialBase
from utils.math_distributions import (lognorm_mean, DeterministicDistribution,
                                      HypoExponentialDistribution) 



class LibraryExponentialLogNormal(LibraryExponentialBase):
    """ represents a single receptor library with random entries. The only
    parameters that characterizes this library is the density of entries. """


    def __init__(self, num_substrates, num_receptors, mean_sensitivity=1,
                 spread=0.1, parameters=None):
        """ represents a theoretical receptor library where the entries of the
        sensitivity matrix are drawn from a log-normal distribution """
        super(LibraryExponentialLogNormal, self).__init__(num_substrates,
                                                          num_receptors,
                                                          parameters)
        self.mean_sensitivity = mean_sensitivity
        self.spread = spread

        
    @property
    def repr_params(self):
        """ return the important parameters that are shown in __repr__ """
        params = super(LibraryExponentialLogNormal, self).repr_params
        params.append('spread=%g' % self.spread)
        params.append('S0=%g' % self.mean_sensitivity)
        return params

        
    @property
    def sens_mat_distribution(self):
        """ returns the probability distribution for the interaction matrix """
        if self.spread == 0:
            return DeterministicDistribution(loc=self.mean_sensitivity)
        else:
            return lognorm_mean(self.mean_sensitivity, self.spread)


    @property
    def init_arguments(self):
        """ return the parameters of the model that can be used to reconstruct
        it by calling the __init__ method with these arguments """
        args = super(LibraryExponentialLogNormal, self).init_arguments
        args['mean_sensitivity'] = self.mean_sensitivity
        args['spread'] = self.spread
        return args


    @classmethod
    def get_random_arguments(cls, **kwargs):
        """ create random arguments for creating test instances """
        parent_cls = super(LibraryExponentialLogNormal, cls)
        args = parent_cls.get_random_arguments(**kwargs)
        S0 = np.random.random() + 0.5
        args['mean_sensitivity'] = kwargs.get('mean_sensitivity', S0)
        args['spread'] = kwargs.get('spread', 0.1*np.random.random())
        return args


#     @classmethod
#     def from_numeric(cls, numeric_model, typical_sensitivity=None, spread=None,
#                      parameters=None):
#         """ creates an instance of this class by using parameters from a related
#         numeric instance """
#         # set parameters
#         kwargs = {'parameters': parameters}
#         if typical_sensitivity is not None:
#             kwargs['mean_sensitivity'] = typical_sensitivity
#         elif numeric_model.sens_mat is not None:
#             kwargs['mean_sensitivity'] = numeric_model.sens_mat.mean()
#         if spread is not None:
#             kwargs['spread'] = spread
#         
#         # create the object
#         obj = cls(numeric_model.Ns, numeric_model.Nr, **kwargs)
#         
#         # copy the commonness from the numeric model
#         obj.concentration = numeric_model.commonness
#         
#         return obj


    def receptor_activity(self):
        """ return the probability with which a single receptor is activated 
        by typical mixtures """
        p_i = self.concentrations
        
        if self.spread == 0:
            # simple case in which the interaction matrix elements are the same:
            #     I_ai = self.mean_sensitivity
            # this is the limiting case
            if self.is_homogeneous_mixture:
                # evaluate the full integral for the case where all substrates
                # are equally likely
                dist = stats.gamma(a=self.Ns, scale=p_i)
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
            if self.is_homogeneous_mixture:
                # FIXME: this is the result for the simple case where all
                # I_ai are equal for a given a
                dist = stats.gamma(a=self.Ns, scale=p_i[0])
                cdf = self.sens_mat_distribution.cdf
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
    
    
    def receptor_activity_estimate(self, method='normal'):
        """ return the probability with which a single receptor is activated 
        by typical mixtures using an gaussian approximation """
        p_i = self.concentrations

        if method == 'normal':
            # use a normal distribution for approximations

            if self.spread == 0:
                # simple case in which all matrix elements are the same:
                #     I_ai = self.mean_sensitivity
                # this is the limiting case
    
                # get the moments of the hypoexponential distribution
                ctot_mean = p_i.sum()
                ctot_var = np.sum(p_i**2)
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
                S0 = self.mean_sensitivity
                sigma2 = self.spread**2
                
                # we first determine the mean and the variance of the
                # distribution of z = I_ai * c_i, which is the distribution for
                # a single matrix element I_ai multiplied the concentration of a
                # single substrate
                zi_mean = p_i*S0 * np.exp(0.5*sigma2)
                zi_var = (S0*p_i)**2 * (2*np.exp(sigma2) - 1) * np.exp(sigma2)
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
             
            if self.spread == 0:
                # simple case in which the matrix elements are the same:
                #     I_ai = self.mean_sensitivity
                # this is the limiting case
    
                # get the moments of the hypoexponential distribution
                ctot_mean = p_i.sum()
                ctot_var = np.sum(p_i**2)
                
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
                S0 = self.mean_sensitivity
                sigma2 = self.spread**2
                
                # we first determine the mean and the variance of the
                # distribution of z = I_ai * c_i, which is the distribution for
                # a single matrix element I_ai multiplied the concentration of a
                # single substrate
                c_mean = p_i.mean()
                z_mean = S0*c_mean * np.exp(0.5*sigma2)
                z_var = (S0*c_mean)**2 * (2*np.exp(sigma2) - 1) * np.exp(sigma2)
                
                # calculate the parameters of the associated gamma distribution
                alpha = z_mean**2 / z_var
                beta = z_var / z_mean
                
                # add up all the N_s distributions to find the probability
                # distribution for determining the activity of a receptor
                alpha *= self.Ns
                # this assumes that beta is the same for all individual
                # substrates, which is only the case for homogeneous mixtures
                if not self.is_homogeneous_mixture:
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


    def get_optimal_library(self):
        """ returns an estimate for the optimal parameters for the random
        interaction matrices """
        sigma = np.pi / np.sqrt(6)
        c_mean = self.concentration_means.mean()
        S0 = np.exp(-0.5 * sigma**2)/(self.Ns * c_mean)
        return {'distribution': 'log_normal',
                'mean_sensitivity': S0, 'spread': sigma}


    def get_optimal_sigma(self):    
        """ estimate the optimal width of the log-normal distribution """
        return np.pi / np.sqrt(6)


    def get_optimal_typical_sensitivity(self, estimate=None, approximation=None):
        """ estimates the optimal average value of the interaction matrix
        elements """  
        if estimate is None:
            c_mean = self.concentration_means.mean()
            estimate = np.exp(-0.5 * self.spread**2)/(self.Ns * c_mean)
        
        # find best mean_sensitivity by optimizing until the average receptor
        # activity is 0.5
        obj = self.copy() #< copy of the current object for optimization
        
        # check which approximation to use
        if approximation is None or approximation == 'none':
            # optimize using true activity calculations
            result = None
            def opt_goal(S0):
                """ helper function to find optimum numerically """ 
                obj.mean_sensitivity = S0
                return 0.5 - obj.receptor_activity()
            
        elif approximation == 'estimate':
            # do not do any numerical optimization
            result = estimate
            
        else:
            # optimize using approximate activity estimates
            result = None
            def opt_goal(S0):
                """ helper function to find optimum numerically """ 
                obj.mean_sensitivity = S0
                return 0.5 - obj.receptor_activity_estimate(approximation)

        if result is None:        
            try:
                result = optimize.newton(opt_goal, estimate)
            except RuntimeError:
                result = np.nan
            
        return result 
            
            
    def mutual_information(self, approximation=None):
        """ return a theoretical estimate of the mutual information between
        input and output """
        if approximation is None or approximation == 'none':
            q_n = self.receptor_activity()
        else:
            q_n = self.receptor_activity_estimate(approximation)
            
        if q_n == 0 or q_n == 1:
            # receptors are never or always activated
            return 0

        else:
            # calculate the information a single receptor contributes            
            H_r = -(q_n*np.log2(q_n) + (1 - q_n)*np.log2(1 - q_n))
            # calculate the MI assuming that receptors are independent
            MI = self.Ns - self.Ns*(1 - H_r/self.Ns)**self.Nr
            return MI
        
        