'''
Created on Dec 29, 2015

@author: David Zwicker <dzwicker@seas.harvard.edu>
'''

from __future__ import division, absolute_import

import numpy as np

from binary_response.sparse_mixtures.lib_spr_numeric import LibrarySparseNumeric
from .pc_base import PrimacyCodingMixin
from utils.misc import nlargest_indices, take_popcount



class PrimacyCodingNumeric(PrimacyCodingMixin, LibrarySparseNumeric):
    """ represents a single receptor library that handles sparse mixtures that
    encode their signal using the `coding_receptors` most active recpetors """
            
            
    #===========================================================================
    # OVERWRITE METHODS OF THE BINARY RESPONSE MODEL
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
        raise NotImplementedError
 
        
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
        raise NotImplementedError
                
        