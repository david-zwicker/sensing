'''
Created on Apr 1, 2015

@author: David Zwicker <dzwicker@seas.harvard.edu>
'''

from __future__ import division

import numpy as np

from ..library_base import LibraryBase



class LibraryContinuousBase(LibraryBase):
    """ represents a single receptor library. This is a base class that provides
    general functionality and parameter management.
    
    For instance, the class provides a framework for calculating ensemble
    averages, where each time new commonness vectors are chosen randomly
    according to the parameters of the last call to `set_commonness`.  
    """

    # default parameters that are used to initialize a class if not overwritten
    parameters_default = {
        'commonness_vector': None,     #< chosen substrate commonness
        'commonness_parameters': None, #< parameters for substrate commonness
    }


    def __init__(self, num_substrates, num_receptors, parameters=None):
        """ initialize a receptor library by setting the number of receptors,
        the number of substrates it can respond to, and optional additional
        parameters in the parameter dictionary """
        super(LibraryContinuousBase, self).__init__(num_substrates,
                                                    num_receptors,
                                                    parameters)

        # apply the parameters to the object
        if self.parameters['commonness_parameters'] is None:
            self.commonness = self.parameters['commonness_vector']
        else:
            self.set_commonness(**self.parameters['commonness_parameters'])


    @classmethod
    def get_random_arguments(cls, **kwargs):
        """ create random arguments for creating test instances """
        Ns = kwargs.get('Ns', np.random.randint(3, 6))
        Nr = kwargs.get('Nr', np.random.randint(2, 4))
        parameters = {'commonness_vector': np.random.random(Ns)}
        return [Ns, Nr, parameters]


    @property
    def commonness(self):
        """ return the commonness vector """
        return self._hs
    
    @commonness.setter
    def commonness(self, hs):
        """ sets the commonness and the associated substrate probability """
        if hs is None:
            # initialize with default values, but don't save the parameters
            self._hs = -np.ones(self.Ns)
            
        else:
            if len(hs) != self.Ns:
                raise ValueError('Length of the commonness vector must match '
                                 'the number of substrates.')
            if any(hs >= 0):
                raise ValueError('Commonness vector must only contain negative '
                                 'entries.')
            self._hs = np.asarray(hs)
            
            # save the values, since they were set explicitly 
            self.parameters['commonness_vector'] = self._hs
    
    
    @property
    def is_homogeneous(self):
        """ returns True if the mixture is homogeneous """
        h_i = self.commonness
        return np.allclose(h_i, h_i[0])
            
    
    def set_commonness(self, scheme='const', mean_concentration=1, **kwargs):
        """ picks a commonness vector according to the supplied parameters:
        `mean_concentration` sets the mean concentration to expect for each
            individual substrate
        """
        if scheme == 'const':
            # all substrates are equally likely
            hs = np.full(self.Ns, -1/mean_concentration)
            
        else:
            raise ValueError('Unknown commonness scheme `%s`' % scheme)
        
        self.commonness = hs
        
        # we additionally store the parameters that were used for this function
        c_params = {'scheme': scheme, 'mean_concentration': mean_concentration}
        c_params.update(kwargs)
        self.parameters['commonness_parameters'] = c_params  


