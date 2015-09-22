'''
Created on May 1, 2015

@author: zwicker
'''

from __future__ import division

import numpy as np
from scipy import special

from utils.misc import xlog2x

from .lib_spr_base import LibrarySparseBase  # @UnresolvedImport
from ..binary_mixtures.lib_bin_numeric import _sample_binary_mixtures
from ..library_numeric_base import (LibraryNumericMixin,
                                    optimize_continuous_library,
                                    get_sensitivity_matrix)



class LibrarySparseNumeric(LibraryNumericMixin, LibrarySparseBase):
    """ represents a single receptor library that handles sparse mixtures """

    # default parameters that are used to initialize a class if not overwritten
    parameters_default = {
        'max_num_receptors': 28,           #< prevents memory overflows
        'max_steps': 1e7,                  #< maximal number of steps 
        'sensitivity_matrix': None,        #< will be calculated if not given
        'sensitivity_matrix_params': None, #< parameters determining I_ai
        'fixed_mixture_size': None,     #< fixed m or None
        'monte_carlo_steps': 'auto',       #< default steps for monte carlo
        'monte_carlo_steps_min': 1e4,      #< minimal steps for monte carlo
        'monte_carlo_steps_max': 1e5,      #< maximal steps for monte carlo
    }
         
            
    @property
    def monte_carlo_steps(self):
        """ calculate the number of monte carlo steps to do """
        if self.parameters['monte_carlo_steps'] == 'auto':
            steps_min = self.parameters['monte_carlo_steps_min']
            steps_max = self.parameters['monte_carlo_steps_max']
            steps = np.clip(10 * 2**self.Nr, steps_min, steps_max) 
            # Here, the factor 10 is an arbitrary scaling factor
        else:
            steps = self.parameters['monte_carlo_steps']
            
        return int(steps)

            
    @classmethod
    def create_test_instance(cls, **kwargs):
        """ creates a test instance used for consistency tests """
        parent = super(LibrarySparseNumeric, cls)
        obj_base = parent.create_test_instance(**kwargs)

        # determine optimal parameters for the sensitivity matrix
        from binary_response.sparse_mixtures.lib_spr_theory import LibrarySparseLogNormal  
        theory = LibrarySparseLogNormal.from_other(obj_base, spread=3) 
        
        obj = cls.from_other(obj_base)
        obj.choose_sensitivity_matrix(**theory.get_optimal_library())
        return obj
    

    def choose_sensitivity_matrix(self, distribution, mean_sensitivity=1,
                                  **kwargs):
        """ chooses the sensitivity matrix """
        self.sens_mat, sens_mat_params = get_sensitivity_matrix(
                  self.Nr, self.Ns, distribution, mean_sensitivity, **kwargs)

        # save the parameters determining this matrix
        self.parameters['sensitivity_matrix_params'] = sens_mat_params

    choose_sensitivity_matrix.__doc__ = get_sensitivity_matrix.__doc__  


    @property
    def _sample_steps(self):
        """ returns the number of steps that are sampled """
        return self.monte_carlo_steps


    def _sample_mixtures(self, steps=None):
        """ sample mixtures with uniform probability yielding single mixtures """
        
        if steps is None:
            steps = self._sample_steps
        
        d_i = self.concentrations
        
        for b in _sample_binary_mixtures(self, steps=steps, dtype=np.bool):
            # boolean b vector is True for the ligands that are present
            
            # choose concentrations for the ligands
            c = np.random.exponential(size=self.Ns) * d_i
            
            # set concentration of ligands that are not present to zero 
            c[~b] = 0
            
            yield c

        
    def concentration_statistics_estimate(self):
        """ estimate the statistics for each individual substrate """
        return super(LibrarySparseNumeric, self).concentration_statistics()
        
        
    def mutual_information_estimate_fast(self):
        """ returns a simple estimate of the mutual information for the special
        case that ret_prob_activity=False, excitation_model='default',
        mutual_information_method='default', and clip=True.
        """
        pi = self.substrate_probabilities
        di = self.concentrations
        ci_mean = di * pi
        ci_var = di * ci_mean * pi*(2 - pi)
        
        # calculate statistics of e_n = \sum_i S_ni * c_i        
        S_ni = self.sens_mat
        en_mean = np.dot(S_ni, ci_mean)
        enm_cov = np.einsum('ni,mi,i->nm', S_ni, S_ni, ci_var)
        en_var = np.diag(enm_cov)
        en_std = np.sqrt(en_var)

        with np.errstate(divide='ignore', invalid='ignore'):
            # calculate the receptor activity
            en_cv2 = en_var / en_mean**2
            enum = np.log(np.sqrt(1 + en_cv2) / en_mean)
            denom = np.sqrt(2*np.log(1 + en_cv2))
            q_n = 0.5 * special.erfc(enum/denom)
        
            # calculate the receptor crosstalk
            rho = np.divide(enm_cov, np.outer(en_std, en_std))

        # replace values that are nan with zero. This might not be exact,
        # but only occurs in corner cases that are not interesting to us
        idx = ~np.isfinite(q_n)
        if np.any(idx):
            q_n[idx] = (en_mean >= 1)
        rho[np.isnan(rho)] = 0
            
        # estimate the crosstalk
        q_nm = rho / (2*np.pi)
        
        # calculate the approximate mutual information
        MI = -np.sum(xlog2x(q_n) + xlog2x(1 - q_n))
    
        # calculate the crosstalk
        MI -= 8/np.log(2) * np.sum(np.triu(q_nm, 1)**2)
        
        return np.clip(MI, 0, self.Nr)
                
        
    def optimize_library(self, target, **kwargs):
        """ optimizes the current library to maximize the result of the target
        function using gradient descent. By default, the function returns the
        best value and the associated sensitivity matrix as result.        
        """
        return optimize_continuous_library(self, target, **kwargs)
        
           
    