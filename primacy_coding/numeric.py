'''
Created on Dec 29, 2015

@author: David Zwicker <dzwicker@seas.harvard.edu>
'''

from __future__ import division, absolute_import

import numpy as np
from scipy import misc

from binary_response.sparse_mixtures.lib_spr_numeric import LibrarySparseNumeric
from utils.misc import nlargest_indices, take_popcount



class PrimacyCodingNumeric(LibrarySparseNumeric):
    """ represents a single receptor library that handles sparse mixtures """

    # default parameters that are used to initialize a class if not overwritten
    parameters_default = {
        'coding_receptors': 1, 
    }
    

    def __init__(self, num_substrates, num_receptors, parameters=None):
        """ initialize a receptor library by setting the number of receptors,
        the number of substrates it can respond to, and optional additional
        parameters in the parameter dictionary.
        `coding_receptors` is the number of receptors with the largest
            excitation, which are used for coding
        """
        super(PrimacyCodingNumeric, self).__init__(num_substrates,
                                                   num_receptors, parameters)


    @property
    def repr_params(self):
        """ return the important parameters that are shown in __repr__ """
        params = super(PrimacyCodingNumeric, self).repr_params
        params.insert(2, 'Nr_k=%d' % self.coding_receptors)
        return params
    
    
    @classmethod
    def get_random_arguments(cls, coding_receptors=None, **kwargs):
        """ create random args for creating test instances """
        args = super(PrimacyCodingNumeric, cls).get_random_arguments(**kwargs)
        
        if coding_receptors is None:
            coding_receptors = np.random.randint(1, 3)
            
        args['parameters']['coding_receptors'] = coding_receptors

        return args
    
    
    @property
    def coding_receptors(self):
        """ return the number of receptors used for coding """
        return self.parameters['coding_receptors']
    
    
    @coding_receptors.setter
    def coding_receptors(self, Nr_k):
        """ set the number of receptors used for coding """
        self.parameters['coding_receptors'] = Nr_k


    @property
    def mutual_information_max(self):
        """ returns an upper bound to the mutual information """
        return np.log2(misc.comb(self.Nr, self.coding_receptors))
  
            
    #===========================================================================
    # OVERWRITE METHODS THAT CALCULATE ACTIVITY PATTERNS AND INFORMATION
    #===========================================================================


    def receptor_activity_monte_carlo(self, ret_correlations=False):
        """ calculates the average activity of each receptor """
        # prevent integer overflow in collecting activity patterns
        assert self.Nr <= self.parameters['max_num_receptors'] <= 63

        S_ni = self.sens_mat

        r_n = np.zeros(self.Nr)
        if ret_correlations:
            r_nm = np.zeros((self.Nr, self.Nr))
        
        for c_i in self._sample_mixtures():
            e_n = np.dot(S_ni, c_i)
            a_ni = nlargest_indices(e_n, self.coding_receptors)
            r_n[a_ni] += 1
            if ret_correlations:
                r_nm[a_ni[:, None], a_ni[None, :]] += 1
            
        r_n /= self._sample_steps
        if ret_correlations:
            r_nm /= self._sample_steps
            return r_n, r_nm
        else:
            return r_n


    def receptor_activity_estimate(self, ret_correlations=False,
                                   excitation_model='default', clip=False):
        """ estimates the average activity of each receptor """
        raise NotImplementedError()
#         en_stats = self.excitation_statistics_estimate()
# 
#         # calculate the receptor activity
#         r_n = self._estimate_qn_from_en(en_stats,
#                                         excitation_model=excitation_model)
#         if clip:
#             np.clip(r_n, 0, 1, r_n)
# 
#         if ret_correlations:
#             # calculate the correlated activity 
#             q_nm = self._estimate_qnm_from_en(en_stats)
#             r_nm = np.outer(r_n, r_n) + q_nm
#             if clip:
#                 np.clip(r_nm, 0, 1, r_nm)
# 
#             return r_n, r_nm
#         else:
#             return r_n   
 
        
    def receptor_crosstalk_estimate(self, ret_receptor_activity=False,
                                    excitation_model='default', clip=False):
        """ calculates the average activity of the receptor as a response to 
        single ligands. """
        en_stats = self.excitation_statistics_estimate()

        # calculate the receptor crosstalk
        q_nm = self._estimate_qnm_from_en(en_stats)
        if clip:
            np.clip(q_nm, 0, 1, q_nm)

        if ret_receptor_activity:
            # calculate the receptor activity
            q_n = self._estimate_qn_from_en(en_stats, excitation_model)
            if clip:
                np.clip(q_n, 0, 1, q_n)

            return q_n, q_nm
        else:
            return q_nm


    def mutual_information_monte_carlo(self, ret_prob_activity=False):
        """ calculate the mutual information using a monte carlo strategy. The
        number of steps is given by the model parameter 'monte_carlo_steps' """
        # prevent integer overflow in collecting activity patterns
        assert self.Nr <= self.parameters['max_num_receptors'] <= 63

        base = 2 ** np.arange(0, self.Nr)

        # sample mixtures according to the probabilities of finding
        # substrates
        count_a = np.zeros(2**self.Nr)
        for c in self._sample_mixtures():
            # get the excitation vector ...
            e_n = np.dot(self.sens_mat, c)
            
            # ... determine the activity ...
            a_ni = nlargest_indices(e_n, self.coding_receptors)
            
            # ... and represent it as a single integer
            a_id = base[a_ni].sum()
            
            # increment counter for this output
            count_a[a_id] += 1
            
        # count_a contains the number of times output pattern a was observed.
        # We can thus construct P_a(a) from count_a. 
        q_n = count_a / count_a.sum()
        
        # calculate the mutual information from the result pattern
        MI = -sum(q*np.log2(q) for q in q_n if q != 0)

        if ret_prob_activity:
            return MI, take_popcount(q_n, self.coding_receptors)
        else:
            return MI

        
    def mutual_information_estimate_fast(self):
        """ returns a simple estimate of the mutual information for the special
        case that ret_prob_activity=False, excitation_model='default',
        mutual_information_method='default', and clip=True.
        """
        raise NotImplementedError()
#     
#         pi = self.substrate_probabilities
#         c_means = self.c_means
#         
#         ci_mean = pi * c_means
#         ci_var = pi * ((1 - pi)*c_means**2 + self.c_vars)
#         
#         # calculate statistics of e_n = \sum_i S_ni * c_i        
#         S_ni = self.sens_mat
#         en_mean = np.dot(S_ni, ci_mean)
#         enm_cov = np.einsum('ni,mi,i->nm', S_ni, S_ni, ci_var)
#         en_var = np.diag(enm_cov)
#         en_std = np.sqrt(en_var)
# 
#         with np.errstate(divide='ignore', invalid='ignore'):
#             # calculate the receptor activity
#             en_cv2 = en_var / en_mean**2
#             enum = np.log(np.sqrt(1 + en_cv2) / en_mean)
#             denom = np.sqrt(2*np.log1p(en_cv2))
#             q_n = 0.5 * special.erfc(enum/denom)
#         
#             # calculate the receptor crosstalk
#             rho = np.divide(enm_cov, np.outer(en_std, en_std))
# 
#         # replace values that are nan with zero. This might not be exact,
#         # but only occurs in corner cases that are not interesting to us
#         idx = ~np.isfinite(q_n)
#         if np.any(idx):
#             q_n[idx] = (en_mean >= 1)
#         rho[np.isnan(rho)] = 0
#             
#         # estimate the crosstalk
#         q_nm = rho / (2*np.pi)
#         
#         # calculate the approximate mutual information
#         MI = -np.sum(xlog2x(q_n) + xlog2x(1 - q_n))
#     
#         # calculate the crosstalk
#         MI -= 8/np.log(2) * np.sum(np.triu(q_nm, 1)**2)
#         
#         return np.clip(MI, 0, self.Nr)
                
        