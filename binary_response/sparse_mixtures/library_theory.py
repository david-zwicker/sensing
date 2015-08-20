'''
Created on Mar 31, 2015

@author: David Zwicker <dzwicker@seas.harvard.edu>
'''

from __future__ import division

import numpy as np

from utils.math_distributions import lognorm_mean, DeterministicDistribution
from .library_base import LibrarySparseBase  # @UnresolvedImport



class LibrarySparseTheoryBase(LibrarySparseBase):
    """ base class for theoretical libraries for sparse mixtures """
    
    def receptor_activity(self, ret_correlations=False, clip=True):
        """ return the probability with which a single receptor is activated 
        by typical mixtures """
        q_n, q_nm = self.receptor_crosstalk(ret_receptor_activity=True,
                                            clip=False)
        
        r_n = q_n
        r_nm = q_n**2 + q_nm
        
        if clip:
            r_n = np.clip(r_n, 0, 1)
            r_nm = np.clip(r_nm, 0, 1)
        
        if ret_correlations:
            return r_n, r_nm
        else:
            return r_n
        
        
    def receptor_crosstalk(self, ret_receptor_activity=False, clip=True):
        """ calculates the average activity of the receptor as a response to 
        single ligands. """
        en_mean, en_var, enm_covar = self.excitation_statistics()
        
        if en_mean == 0:
            q_n, q_nm = 0, 0
            
        else:
            q_n = self._estimate_qn_from_en(en_mean, en_var)
            q_nm = self._estimate_qnm_from_en(en_var, enm_covar)
                
            if clip:
                q_n = np.clip(q_n, 0, 1)
                q_nm = np.clip(q_nm, 0, 1)

        if ret_receptor_activity:
            return q_n, q_nm
        else:
            return q_nm


    def mutual_information(self, clip=False):
        """ calculates the typical mutual information """
        q_n, q_nm = self.receptor_crosstalk(ret_receptor_activity=True)
        MI = self._estimate_mutual_information_from_q_stats(q_n, q_nm)
        if clip:
            MI = np.clip(MI, 0, self.Nr)
        return MI



class LibrarySparseBinary(LibrarySparseTheoryBase):
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
    def repr_params(self):
        """ return the important parameters that are shown in __repr__ """
        params = super(LibrarySparseBinary, self).repr_params
        params.append('xi=%g' % self.density)
        params.append('S0=%g' % self.typical_sensitivity)
        return params


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


    def excitation_statistics(self):
        """ calculates the statistics of the excitation of the receptors.
        Returns the mean exciation, the variance, and the covariance matrix """
        if self.correlated_mixture:
            raise NotImplementedError('Not implemented for correlated mixtures')
        
        di = self.concentrations
        pi = self.substrate_probabilities
        S0 = self.typical_sensitivity
        xi = self.density

        # calculate statistics of the sum s_n = S_ni * c_i        
        en_mean = S0 * xi * np.sum(di * pi)
        en_var = S0**2 * xi * np.sum(di**2 * pi*(2 - pi))
        enm_covar = S0**2 * xi**2 * np.sum(di**2 * pi*(2 - pi))
                
        return en_mean, en_var, enm_covar

        
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
        S0 = 1 / (m*d*density + d*np.log(2))
        return {'distribution': 'binary',
                'typical_sensitivity': S0, 'density': density}



class LibrarySparseLogNormal(LibrarySparseTheoryBase):
    """ represents a single receptor library with random entries drawn from a
    lognormal distribution """


    def __init__(self, num_substrates, num_receptors, sigma=1,
                 typical_sensitivity=1, parameters=None):
        """ initialize the receptor library by setting the number of receptors,
        the number of substrates it can respond to, the width of the
        distribution `sigma`, and the typical sensitivity or magnitude S0 of the
        sensitivity matrix """
        super(LibrarySparseLogNormal, self).__init__(num_substrates,
                                                     num_receptors, parameters)
        self.sigma = sigma
        self.typical_sensitivity = typical_sensitivity


    @property
    def repr_params(self):
        """ return the important parameters that are shown in __repr__ """
        params = super(LibrarySparseLogNormal, self).repr_params
        params.append('sigma=%g' % self.sigma)
        params.append('S0=%g' % self.typical_sensitivity)
        return params


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


    def excitation_statistics(self):
        """ calculates the statistics of the excitation of the receptors.
        Returns the mean exciation, the variance, and the covariance matrix """
        if self.correlated_mixture:
            raise NotImplementedError('Not implemented for correlated mixtures')
        
        di = self.concentrations
        pi = self.substrate_probabilities
        S0 = self.typical_sensitivity
        sigma2 = self.sigma ** 2
        
        # calculate statistics of the sum s_n = S_ni * c_i        
        en_mean = S0 * np.sum(di * pi)
        en_var = S0**2 * np.exp(sigma2) * np.sum(di**2 * pi*(2 - pi))
        enm_covar = S0**2 * np.sum(di**2 * pi*(2 - pi))

        return en_mean, en_var, enm_covar
        

    def get_optimal_library(self):
        """ returns an estimate for the optimal parameters for the random
        interaction matrices """
        if self.correlated_mixture:
            raise NotImplementedError('Not implemented for correlated mixtures')

        c_stats = self.concentration_statistics()
        ctot_mean = c_stats['mean'].sum()
        ctot_var = c_stats['var'].sum()

        sigma_opt = 2
        S0_opt = np.sqrt(1/ctot_mean**2
                         + ctot_var/ctot_mean**4 * np.exp(sigma_opt**2)) 
        
        return {'distribution': 'log_normal',
                'typical_sensitivity': S0_opt, 'sigma': sigma_opt}
    
        