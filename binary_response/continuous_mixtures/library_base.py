'''
Created on Apr 1, 2015

@author: David Zwicker <dzwicker@seas.harvard.edu>
'''

from __future__ import division

import numpy as np
from scipy import stats

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
        """ create random args for creating test instances """
        args = super(LibraryContinuousBase, cls).get_random_arguments(**kwargs)
        if kwargs.get('homogeneous_mixture', False):
            hs = np.full(args['num_substrates'], np.random.random() - 1.5)
        else:
            hs = np.random.random(args['num_substrates']) - 1.5
        args['parameters'] = {'commonness_vector': hs}
        return args


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
    
    
    def get_concentration_distribution(self, i):
        """ returns the concentration distribution for component i """
        return stats.expon(scale=-1/self.commonness[i])
    
    
    def get_concentration_means(self):
        """ returns the mean concentration with which each substrate is
        expected """
        return -1/self.commonness
    
    
    @property
    def is_homogeneous(self):
        """ returns True if the mixture is homogeneous """
        h_i = self.commonness
        return np.allclose(h_i, h_i[0])
            
    
    def set_commonness(self, scheme, total_concentration, **kwargs):
        """ picks a commonness vector according to the supplied parameters:
        `total_concentration` sets the total concentration to expect for the
            mixture on average.
        """
        mean_concentration = total_concentration / self.Ns
        
        if scheme == 'const':
            # all substrates are equally likely
            if total_concentration == 0:
                hs = np.full(self.Ns, -np.inf)
            else:
                hs = np.full(self.Ns, -1/mean_concentration)
                
        elif scheme == 'random_uniform':
            # draw the mean probabilities from a uniform distribution
            c_means = np.random.uniform(0, 2*mean_concentration, self.Ns)
            # make sure that the mean concentration is correct
            c_means *= total_concentration/c_means.sum()
            # convert this to commonness values
            hs = -1/c_means
            
        else:
            raise ValueError('Unknown commonness scheme `%s`' % scheme)
        
        self.commonness = hs
        
        # we additionally store the parameters that were used for this function
        c_params = {'scheme': scheme, 'total_concentration': total_concentration}
        c_params.update(kwargs)
        self.parameters['commonness_parameters'] = c_params  

