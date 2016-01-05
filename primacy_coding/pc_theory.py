'''
Created on Jan 5, 2016

@author: David Zwicker <dzwicker@seas.harvard.edu>
'''

from __future__ import division

from binary_response.sparse_mixtures.lib_spr_theory import LibrarySparseLogNormal
from .pc_base import PrimacyCodingMixin



class PrimacyCodingTheory(PrimacyCodingMixin, LibrarySparseLogNormal):
    

    #===========================================================================
    # OVERWRITE METHODS OF THE BINARY RESPONSE MODEL
    #===========================================================================


    def get_optimal_parameters(self):
        raise NotImplementedError


    def receptor_activity(self):
        """ return the probability with which a single receptor is activated 
        by typical mixtures """
        raise NotImplementedError
        
        
    def receptor_crosstalk(self):
        """ calculates the average activity of the receptor as a response to 
        single ligands. """
        raise NotImplementedError


    def mutual_information(self):
        """ calculates the typical mutual information """
        raise NotImplementedError
    