'''
Created on Mar 31, 2015

@author: David Zwicker <dzwicker@seas.harvard.edu>
'''

from __future__ import division

import numpy as np

from utils.math_distributions import (lognorm_mean, loguniform_mean,
                                      DeterministicDistribution)
from .library_base import LibrarySparseBase  # @UnresolvedImport



class LibrarySparseTheoryBase(LibrarySparseBase):
    """ base class for theoretical libraries for sparse mixtures """
    
    
    def sensitivity_stats(self):
        """ returns the statistics of the sensitivity matrix """
        raise NotImplementedError
    
    
    def excitation_statistics(self):
        """ calculates the statistics of the excitation of the receptors.
        Returns the mean exciation, the variance, and the covariance matrix """
        if self.correlated_mixture:
            raise NotImplementedError('Not implemented for correlated mixtures')
        
        ctot_stats = self.ctot_statistics()
        S_stats = self.sensitivity_stats()
        S_mean = S_stats['mean']
        
        # calculate statistics of the sum s_n = S_ni * c_i        
        en_mean = S_mean * ctot_stats['mean']
        en_var = (S_mean**2 + S_stats['var']) * ctot_stats['var']
        enm_covar = (S_mean**2 + S_stats['covar']) * ctot_stats['var']

        return {'mean': en_mean, 'std': np.sqrt(en_var), 'var': en_var,
                'covar': enm_covar}
        
    
    def receptor_activity(self, approx_prob=False, ret_correlations=False,
                          clip=True):
        """ return the probability with which a single receptor is activated 
        by typical mixtures """
        q_n, q_nm = self.receptor_crosstalk(
                approx_prob=approx_prob, ret_receptor_activity=True, clip=False)
        
        r_n = q_n
        r_nm = q_n**2 + q_nm
        
        if clip:
            r_n = np.clip(r_n, 0, 1)
            r_nm = np.clip(r_nm, 0, 1)
        
        if ret_correlations:
            return r_n, r_nm
        else:
            return r_n
        
        
    def receptor_crosstalk(self, approx_prob=False, ret_receptor_activity=False,
                           clip=True):
        """ calculates the average activity of the receptor as a response to 
        single ligands. """
        en_stats = self.excitation_statistics()
        
        if en_stats['mean'] == 0:
            q_n, q_nm = 0, 0
            
        else:
            q_n = self._estimate_qn_from_en(en_stats, approx_prob=approx_prob)
            q_nm = self._estimate_qnm_from_en(en_stats)
                
            if clip:
                q_n = np.clip(q_n, 0, 1)
                q_nm = np.clip(q_nm, 0, 1)

        if ret_receptor_activity:
            return q_n, q_nm
        else:
            return q_nm


    def mutual_information(self, approx_prob=False, use_polynom=True,
                           clip=False):
        """ calculates the typical mutual information """
        # get receptor activity probabilities
        q_n, q_nm = self.receptor_crosstalk(approx_prob=approx_prob,
                                            ret_receptor_activity=True)
        
        # estimate mutual information from this
        MI = self._estimate_mutual_information_from_q_stats(
                                            q_n, q_nm, use_polynom=use_polynom)
        
        if clip:
            return np.clip(MI, 0, self.Nr)
        else:
            return MI
        
        
    def set_optimal_parameters(self, **kwargs):
        """ adapts the parameters of this library to be close to optimal """
        params = self.get_optimal_library()
        del params['distribution']
        self.__dict__.update(params)



class LibrarySparseBinary(LibrarySparseTheoryBase):
    """ represents a single receptor library with random entries. The only
    parameters that characterizes this library is the density of entries and
    their magnitude """


    def __init__(self, num_substrates, num_receptors,
                 mean_sensitivity=1, parameters=None, **kwargs):
        """ initialize the receptor library by setting the number of receptors,
        the number of substrates it can respond to, the fraction `density` of
        substrates a single receptor responds to, and the typical sensitivity
        or magnitude S0 of the sensitivity matrix """
        super(LibrarySparseBinary, self).__init__(num_substrates,
                                                  num_receptors, parameters)

        self.mean_sensitivity = mean_sensitivity
        
        if 'standard_deviation' in kwargs:
            standard_deviation = kwargs.pop('standard_deviation')
            S_mean2 = mean_sensitivity**2
            self.density = S_mean2 / (S_mean2 + standard_deviation**2)
        elif 'density' in kwargs:
            self.density = kwargs.pop('density')
        else:
            standard_deviation = 1
            S_mean2 = mean_sensitivity**2
            self.density = S_mean2 / (S_mean2 + standard_deviation**2)

        # raise an error if keyword arguments have not been used
        if len(kwargs) > 0:
            raise ValueError('The following keyword arguments have not been '
                             'used: %s' % str(kwargs)) 


    @property
    def standard_deviation(self):
        """ returns the standard deviation of the sensitivity matrix """
        return self.mean_sensitivity * np.sqrt(1/self.density - 1)


    @property
    def repr_params(self):
        """ return the important parameters that are shown in __repr__ """
        params = super(LibrarySparseBinary, self).repr_params
        params.append('xi=%g' % self.density)
        params.append('S0=%g' % self.mean_sensitivity)
        return params


    @property
    def init_arguments(self):
        """ return the parameters of the model that can be used to reconstruct
        it by calling the __init__ method with these arguments """
        args = super(LibrarySparseBinary, self).init_arguments
        args['density'] = self.density
        args['mean_sensitivity'] = self.mean_sensitivity
        return args


    @classmethod
    def get_random_arguments(cls, **kwargs):
        """ create random arguments for creating test instances """
        args = super(LibrarySparseBinary, cls).get_random_arguments(**kwargs)
        args['density'] = kwargs.get('density', np.random.random())
        S0 = np.random.random() + 0.5
        args['mean_sensitivity'] = kwargs.get('mean_sensitivity', S0)
        return args


    def sensitivity_stats(self):
        """ returns statistics of the sensitivity distribution """
        S0 = self.mean_sensitivity
        var = S0**2 * (1/self.density - 1)
        return {'mean': S0, 'var': var, 'std': np.sqrt(var), 'covar': 0}


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
        S0 = 1 / (m*d*density + d*np.log(2)) / density
        return {'distribution': 'binary',
                'mean_sensitivity': S0, 'density': density}



class LibrarySparseLogNormal(LibrarySparseTheoryBase):
    """ represents a single receptor library with random entries drawn from a
    lognormal distribution """


    def __init__(self, num_substrates, num_receptors, mean_sensitivity=1,
                 correlation=0, parameters=None, **kwargs):
        """ initialize the receptor library by setting the number of receptors,
        the number of substrates it can respond to, and the typical sensitivity
        or magnitude S0 of the sensitivity matrix.
        The width of the distribution is either set by the parameter `spread` or
        by setting the `standard_deviation`.
        """
        super(LibrarySparseLogNormal, self).__init__(num_substrates,
                                                     num_receptors, parameters)
        
        self.mean_sensitivity = mean_sensitivity
        self.correlation = correlation

        if 'standard_deviation' in kwargs:
            standard_deviation = kwargs.pop('standard_deviation')
            cv = standard_deviation / mean_sensitivity 
            self.spread = np.sqrt(np.log(cv**2 + 1))
        elif 'spread' in kwargs:
            self.spread = kwargs.pop('spread')
        else:
            standard_deviation = 1
            cv = standard_deviation / mean_sensitivity 
            self.spread = np.sqrt(np.log(cv**2 + 1))

        # raise an error if keyword arguments have not been used
        if len(kwargs) > 0:
            raise ValueError('The following keyword arguments have not been '
                             'used: %s' % str(kwargs)) 
            
            
    @property
    def standard_deviation(self):
        """ return the standard deviation of the distribution """
        return self.mean_sensitivity * np.sqrt((np.exp(self.spread**2) - 1))
            

    @property
    def repr_params(self):
        """ return the important parameters that are shown in __repr__ """
        params = super(LibrarySparseLogNormal, self).repr_params
        params.append('S0=%g' % self.mean_sensitivity)
        params.append('spread=%g' % self.spread)
        params.append('correlation=%g' % self.correlation)
        return params


    @property
    def init_arguments(self):
        """ return the parameters of the model that can be used to reconstruct
        it by calling the __init__ method with these arguments """
        args = super(LibrarySparseLogNormal, self).init_arguments
        args['mean_sensitivity'] = self.mean_sensitivity
        args['spread'] = self.spread
        args['correlation'] = self.correlation
        return args


    @classmethod
    def get_random_arguments(cls, **kwargs):
        """ create random arguments for creating test instances """
        args = super(LibrarySparseLogNormal, cls).get_random_arguments(**kwargs)
        args['spread'] = kwargs.get('spread', np.random.random() + 0.5)
        S0 = np.random.random() + 0.5
        args['mean_sensitivity'] = kwargs.get('mean_sensitivity', S0)
        return args


    @property
    def sensitivity_distribution(self):
        """ returns the sensitivity distribution """
        if self.correlation != 0:
            raise NotImplementedError('Cannot return the sensitivity '
                                      'distribution with correlations, yet')
        
        if self.spread == 0:
            return DeterministicDistribution(self.mean_sensitivity)
        else:
            return lognorm_mean(self.mean_sensitivity, self.spread)


    def sensitivity_stats(self):
        """ returns statistics of the sensitivity distribution """
        S0 = self.mean_sensitivity
        var = S0**2 * (np.exp(self.spread**2) - 1)
        covar = S0**2 * (np.exp(self.correlation * self.spread**2) - 1)
        return {'mean': S0, 'var': var, 'covar': covar}


    def get_optimal_parameters(self, fixed_parameter='S0'):
        """ returns an estimate for the optimal parameters for the random
        interaction matrices.
            `fixed_parameter` determines which parameter is kept fixed during
                the optimization procedure
        """
        if self.correlated_mixture:
            raise NotImplementedError('Not implemented for correlated mixtures')

        ctot_stats = self.ctot_statistics()
        ctot_mean = ctot_stats['mean']
        ctot_var = ctot_stats['var']
        ctot_cv2 = ctot_var/ctot_mean**2
        
        if fixed_parameter == 'spread':
            # keep the spread parameter fixed and determine the others 
            spread_opt = self.spread
            
            arg = 1 + ctot_cv2 * np.exp(spread_opt**2)
            S0_opt = np.sqrt(arg) / ctot_mean
            std_opt = S0_opt * np.sqrt(np.exp(spread_opt**2) - 1)
            
        elif fixed_parameter == 'S0':
            # keep the typical sensitivity fixed and determine the other params 
            S0_opt = self.mean_sensitivity
            
            arg = (ctot_mean**2 * self.mean_sensitivity**2 - 1)/ctot_cv2
            spread_opt = np.sqrt(np.log(arg))
            std_opt = self.mean_sensitivity * np.sqrt(arg - 1)
            
        else:
            raise ValueError('Parameter `%s` is unknown or cannot be held '
                             'fixed' % fixed_parameter) 
        
        return {'mean_sensitivity': S0_opt, 'spread': spread_opt,
                'standard_deviation': std_opt}
    
    
    def get_optimal_library(self, fixed_parameter='S0'):
        """ returns an estimate for the optimal parameters for the random
        interaction matrices.
            `fixed_parameter` determines which parameter is kept fixed during
                the optimization procedure
        """
        library_opt = self.get_optimal_parameters(fixed_parameter)
        return {'distribution': 'log_normal', 'spread': library_opt['spread'],
                'mean_sensitivity': library_opt['mean_sensitivity'],
                'correlation': 0}
                


class LibrarySparseLogUniform(LibrarySparseTheoryBase):
    """ represents a single receptor library with random entries drawn from a
    log-uniform distribution """


    def __init__(self, num_substrates, num_receptors, spread=1,
                 mean_sensitivity=1, parameters=None):
        """ initialize the receptor library by setting the number of receptors,
        the number of substrates it can respond to, the width of the
        distribution `spread`, and the typical sensitivity or magnitude S0 of the
        sensitivity matrix """
        super(LibrarySparseLogUniform, self).__init__(num_substrates,
                                                      num_receptors, parameters)
        self.spread = spread
        self.mean_sensitivity = mean_sensitivity


    @property
    def repr_params(self):
        """ return the important parameters that are shown in __repr__ """
        params = super(LibrarySparseLogUniform, self).repr_params
        params.append('spread=%g' % self.spread)
        params.append('S0=%g' % self.mean_sensitivity)
        return params


    @property
    def init_arguments(self):
        """ return the parameters of the model that can be used to reconstruct
        it by calling the __init__ method with these arguments """
        args = super(LibrarySparseLogUniform, self).init_arguments
        args['spread'] = self.spread
        args['mean_sensitivity'] = self.mean_sensitivity
        return args


    @classmethod
    def get_random_arguments(cls, **kwargs):
        """ create random arguments for creating test instances """
        args = super(LibrarySparseLogUniform, cls).get_random_arguments(**kwargs)
        args['spread'] = kwargs.get('spread', np.random.random() + 0.1)
        S0 = np.random.random() + 0.5
        args['mean_sensitivity'] = kwargs.get('mean_sensitivity', S0)
        return args


    @property
    def sensitivity_distribution(self):
        """ returns the sensitivity distribution """
        if self.spread == 0:
            return DeterministicDistribution(self.mean_sensitivity)
        else:
            return loguniform_mean(self.mean_sensitivity, np.exp(self.spread))


    def sensitivity_stats(self):
        """ returns statistics of the sensitivity distribution """
        S0 = self.mean_sensitivity
        spread = self.spread
        
        # calculate the unscaled variance
        if spread == 0:
            var_S1 = 0
        else:
            exp_s2 = np.exp(spread)**2
            var_S1 = (1 - exp_s2 + (1 + exp_s2)*spread)/(exp_s2 - 1)
        return {'mean': S0, 'var': S0**2 * var_S1, 'covar': 0}
    

    def get_optimal_library(self, spread_opt=2):
        """ returns an estimate for the optimal parameters for the random
        interaction matrices """
        if self.correlated_mixture:
            raise NotImplementedError('Not implemented for correlated mixtures')

        ctot_stats = self.ctot_statistics()
        ctot_mean = ctot_stats['mean']
        ctot_var = ctot_stats['var']

        if self.spread == 0:
            term = 1 
        else:
            exp_s2 = np.exp(self.spread)**2
            term = (exp_s2 + 1) * self.spread / (exp_s2 - 1)
        S0_opt = np.sqrt(1 + ctot_var/ctot_mean**2 * term) / ctot_mean 
        
        return {'distribution': 'log_normal',
                'mean_sensitivity': S0_opt, 'spread': spread_opt}
        
        
        