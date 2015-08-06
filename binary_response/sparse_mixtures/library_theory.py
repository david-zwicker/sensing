'''
Created on Mar 31, 2015

@author: David Zwicker <dzwicker@seas.harvard.edu>
'''

from __future__ import division

import numpy as np

from utils.math_distributions import lognorm_mean, DeterministicDistribution
from .library_base import LibrarySparseBase  # @UnresolvedImport


PI2 = np.pi * 2



class LibrarySparseBinary(LibrarySparseBase):
    """ represents a single receptor library with random entries. The only
    parameters that characterizes this library is the density of entries and
    their magnitude """


    def __init__(self, num_substrates, num_receptors, density=1,
                 typical_sensitivity=1, parameters=None):
        """ initialize the receptor library by setting the number of receptors,
        the number of substrates it can respond to, the fraction `density` of
        substrates a single receptor responds to, and the typical sensitivity
        or magnitude S0 of the sensitivity matrix """
        super(LibrarySparseBinary, self).__init__(num_substrates,
                                                  num_receptors, parameters)
        self.density = density
        self.typical_sensitivity = typical_sensitivity


    @property
    def init_arguments(self):
        """ return the parameters of the model that can be used to reconstruct
        it by calling the __init__ method with these arguments """
        args = super(LibrarySparseBinary, self).init_arguments
        args['density'] = self.density
        args['typical_sensitivity'] = self.typical_sensitivity
        return args


    @classmethod
    def get_random_arguments(cls, **kwargs):
        """ create random arguments for creating test instances """
        args = super(LibrarySparseBinary, cls).get_random_arguments(**kwargs)
        args['density'] = kwargs.get('density', np.random.random())
        S0 = np.random.random() + 0.5
        args['typical_sensitivity'] = kwargs.get('typical_sensitivity', S0)
        return args


    def activity_single(self):
        """ return the probability with which a single receptor is activated 
        by typical mixtures """
        raise NotImplementedError
        return 1 - np.prod(1 - self.density * self.substrate_probabilities)

        
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
        if self.correlated_mixture:
            raise NotImplementedError('Not implemented for correlated mixtures')

        if assume_homogeneous:
            # calculate the idealized substrate probability
            m_mean = self.mixture_size_statistics()['mean']
            p0 = m_mean / self.Ns
             
        else:
            # check whether the mixtures are all homogeneous
            if len(np.unique(self.commonness)) > 1:
                raise RuntimeError('The estimate only works for homogeneous '
                                   'mixtures so far.')
            p0 = self.substrate_probabilities.mean()
            
        # calculate the fraction for the homogeneous case
        return (1 - 2**(-1/self.Ns))/p0
    
    
    def get_optimal_library(self):
        """ returns an estimate for the optimal parameters for the random
        interaction matrices """
        if self.correlated_mixture:
            raise NotImplementedError('Not implemented for correlated mixtures')

        m = self.mixture_size_statistics()['mean']
        d = self.concentration_statistics()['mean'].mean()
        density = self.density_optimal()
        I0 = 1 / (m*d*density + d*np.log(2))
        return {'distribution': 'binary',
                'typical_sensitivity': I0, 'density': density}



class LibrarySparseLogNormal(LibrarySparseBase):
    """ represents a single receptor library with random entries drawn from a
    lognormal distribution """


    def __init__(self, num_substrates, num_receptors, sigma=1,
                 typical_sensitivity=1, parameters=None):
        """ initialize the receptor library by setting the number of receptors,
        the number of substrates it can respond to, the width of the
        distribution `sigma`, and the typical sensitivity or magnitude I0 of the
        sensitivity matrix """
        super(LibrarySparseLogNormal, self).__init__(num_substrates,
                                                     num_receptors, parameters)
        self.sigma = sigma
        self.typical_sensitivity = typical_sensitivity


    @property
    def init_arguments(self):
        """ return the parameters of the model that can be used to reconstruct
        it by calling the __init__ method with these arguments """
        args = super(LibrarySparseLogNormal, self).init_arguments
        args['sigma'] = self.sigma
        args['typical_sensitivity'] = self.typical_sensitivity
        return args


    @classmethod
    def get_random_arguments(cls, **kwargs):
        """ create random arguments for creating test instances """
        args = super(LibrarySparseLogNormal, cls).get_random_arguments(**kwargs)
        args['sigma'] = kwargs.get('sigma', np.random.random() + 0.5)
        S0 = np.random.random() + 0.5
        args['typical_sensitivity'] = kwargs.get('typical_sensitivity', S0)
        return args


    @property
    def sensitivity_distribution(self):
        """ returns the sensitivity distribution """
        if self.sigma == 0:
            return DeterministicDistribution(self.typical_sensitivity)
        else:
            return lognorm_mean(self.typical_sensitivity, self.sigma)


    def receptor_activity(self, ret_correlations=False):
        """ return the probability with which a single receptor is activated 
        by typical mixtures """
        q_n, q_nm = self.receptor_crosstalk(ret_receptor_activity=True)
        if ret_correlations:
            r_nm = np.outer(q_n, q_n) + q_nm
            return q_n, r_nm
        else:
            return q_n
        
        
    def receptor_crosstalk(self, ret_receptor_activity=False):
        """ calculates the average activity of the receptor as a response to 
        single ligands. """
        if self.correlated_mixture:
            raise NotImplementedError('Not implemented for correlated mixtures')
        
        di = self.concentrations
        pi = self.substrate_probabilities
        S0 = self.typical_sensitivity
        sigma2 = self.sigma ** 2
        
        sni_means = S0 * di * pi
        sni_vars = (S0*di)**2 * pi * (2*np.exp(sigma2) - pi)
        
        sn_mean = sni_means.sum()
        sn_var = sni_vars.sum()
        snm = np.sum((S0*di)**2 * pi * (2 - pi))
        with np.errstate(divide='ignore', invalid='ignore'):
            rho = snm / sn_var

        with np.errstate(divide='ignore', invalid='ignore'):
            q_n = 0.5 + (sn_mean - 1) / np.sqrt(PI2 * sn_var) 
            q_n += ((5*sn_mean - 9)*np.sqrt(sn_var))/(8*np.sqrt(2*np.pi))
        q_nm = rho / PI2
        
        # np.clip(q_n, 0, 1, q_n)
        # np.clip(q_nm, 0, 1, q_nm)

        if ret_receptor_activity:
            return q_n, q_nm
        else:
            return q_nm


    def mutual_information(self, clip=False):
        """ calculates the typical mutual information """
        q_n, q_nm = self.receptor_crosstalk(ret_receptor_activity=True)
        MI = self._estimate_mutual_information_from_q(q_n, q_nm, averaged=True)
        if clip:
            np.clip(MI, 0, self.Nr, MI)
        return MI


    def get_optimal_library(self):
        """ returns an estimate for the optimal parameters for the random
        interaction matrices """
        if self.correlated_mixture:
            raise NotImplementedError('Not implemented for correlated mixtures')

        m = self.mixture_size_statistics()['mean']
        d = self.concentration_statistics()['mean'].mean()
        S0 = 1/(m*d)
        return {'distribution': 'log_normal',
                'typical_sensitivity': S0, 'sigma': 1}
    
        