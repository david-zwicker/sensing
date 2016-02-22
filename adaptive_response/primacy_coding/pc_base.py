'''
Created on Jan 5, 2016

@author: David Zwicker <dzwicker@seas.harvard.edu>
'''

from __future__ import division

import numpy as np
from scipy import misc



class PrimacyCodingMixin(object):

    # default parameters that are used to initialize a class if not overwritten
    parameters_default = {
        'coding_receptors': 1, 
    }

    @property
    def repr_params(self):
        """ return the important parameters that are shown in __repr__ """
        params = super(PrimacyCodingMixin, self).repr_params
        params.insert(2, 'Nc=%d' % self.coding_receptors)
        return params
    
    
    @classmethod
    def get_random_arguments(cls, coding_receptors=None, **kwargs):
        """ create random args for creating test instances """
        args = super(PrimacyCodingMixin, cls).get_random_arguments(**kwargs)
        
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
  
