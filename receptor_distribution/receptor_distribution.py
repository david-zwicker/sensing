'''
Created on Jul 15, 2016

@author: David Zwicker <dzwicker@seas.harvard.edu>
'''

from __future__ import division

import numpy as np
from scipy import optimize, spatial



class ExcitationProfiles(object):
    """ class that handles the excitation profiles that describe the input """
    
    def __init__(self, Nr, num_x=16):
        """ initialize the distribution of the excitations of `Nr` receptors
        and `num_x` spatial discretization points """ 
        self.Nr = Nr
        self.num_x = num_x
        self.xs = np.linspace(0, 1, self.num_x)
        
        self.distribution = np.zeros((self.Nr, self.num_x), np.double)


    def choose_exponential_excitations(self, width_prefactor=1, width_decay=1):
        """ choose random, exponentially distributed excitations """
        # choose random parameters
        prefactors = np.random.lognormal(sigma=width_prefactor, size=self.Nr)
        decays = np.random.lognormal(sigma=width_decay, size=self.Nr)
        self.distribution[:] = (prefactors[:, None]
                                * np.exp(-decays[:, None] * self.xs))
        


class ReceptorDistribution(object):
    """ class that describes the distribution of receptors """
    
    cost_weight_total = 1 #< weight of the total excitation in the cost function 
    
    
    def __init__(self, Nr, num_x=16):
        """ initialize the distribution of `Nr` receptors and `num_x` spatial
        discretization points """ 
        self.Nr = Nr
        self.num_x = num_x
        self.xs = np.linspace(0, 1, self.num_x)
        
        # initialize with uniform distribution
        self.distribution = np.empty((self.Nr, self.num_x), np.double)
        self.set_uniform_distribution()

        
    def set_uniform_distribution(self):
        """ sets a homogeneous receptor distribution """
        self.distribution[:] = 1 / self.Nr
        
        
    def _distribution_real2reduced(self, dist_real, out=None):
        """ converts the real receptor distribution to a reduced form """
        if out is None:
            # create a new output array
            out = np.empty((self.Nr - 1, self.num_x), np.double)
        else:
            # create a view of the output array with the right shape
            out = out.reshape((self.Nr - 1, self.num_x))

        a = 1
        for n in range(self.Nr - 1):
            out[n, :] = dist_real[n, :] / a
            a -= dist_real[n, :]
        return out
    
    
    def _distribution_reduced2real(self, dist_reduced, out=None):
        """ converts the reduced receptor distribution to the real one """
        if out is None:
            # create a new output array
            out = np.empty((self.Nr, self.num_x))
        else:
            # create a view of the output array with the right shape
            out = out.reshape((self.Nr, self.num_x))

        # get view of the input array with the right shape
        dist_reduced = dist_reduced.reshape((self.Nr - 1, self.num_x))
        a = 1
        for n in range(self.Nr - 1):
            out[n, :] = a * dist_reduced[n, :]
            a *= 1 - dist_reduced[n, :]
        out[-1, :] = a
        return out
    
    
    @property
    def distribution_reduced(self):
        return self._distribution_real2reduced(self.distribution)

    
    @distribution_reduced.setter
    def distribution_reduced(self, dist_reduced):
        return self._distribution_reduced2real(dist_reduced,
                                               out=self.distribution)
    
    
    def performance(self, excitations):
        """ evaluates the receptor performance for the given excitation """
        e_tot = np.einsum('ij,ij->i', self.distribution,
                          excitations.distribution)
        
        return (self.cost_weight_total * e_tot.sum()
                - spatial.distance.pdist(e_tot[:, None]).sum())
    

    def optimize(self, excitations, method='scipy'):
        """ optimizes the receptor distribution for the given excitations """
        assert excitations.Nr == self.Nr
        assert excitations.num_x == self.num_x
        
        # prepare initial condition
        dist_reduced = self._distribution_real2reduced(self.distribution)
        
        # define the objective function
        def objective_function(dist_reduced):
            self.distribution_reduced = dist_reduced
            return -self.performance(excitations)

        # perform the constraint optimization
        if method == 'scipy':
            bounds = np.array([[0, 1]] * dist_reduced.size)
            opt = optimize.minimize(objective_function, dist_reduced.ravel(),
                                    bounds=bounds)
            self.distribution_reduced = opt.x
            
        elif method == 'cma':
            # use Covariance Matrix Adaptation Evolution Strategy algorithm
            try:
                import cma  # @UnresolvedImport
            except ImportError:
                raise ImportError('The module `cma` is not available. Please '
                                  'install it using `pip install cma` or '
                                  'choose a different optimization method.')

            options = {'bounds': [0, 1],
                       'maxfevals': 5000,
                       'verb_disp': 0,
                       'verb_log': 0}
            
            opt = cma.fmin(objective_function, dist_reduced.ravel(), 1/self.Nr,
                           options=options)
            self.distribution_reduced = opt[0]
            
        else:
            raise ValueError('Unknown optimization method `%s`' % method)
        
        # return the result
        return self.distribution
