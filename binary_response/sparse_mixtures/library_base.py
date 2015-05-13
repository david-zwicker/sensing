'''
Created on Apr 1, 2015

@author: David Zwicker <dzwicker@seas.harvard.edu>
'''

from __future__ import division

import numpy as np
from scipy import stats

from ..binary_mixtures.library_base import LibraryBinaryBase



class LibrarySparseBase(LibraryBinaryBase):
    """ represents a single receptor library. This is a base class that provides
    general functionality and parameter management.
    
    For instance, the class provides a framework for calculating ensemble
    averages, where each time new commonness vectors are chosen randomly
    according to the parameters of the last call to `set_commonness`.  
    """

    # default parameters that are used to initialize a class if not overwritten
    parameters_default = {
        'concentration_vector': None,     #< chosen substrate concentrations
        'concentration_parameters': None, #< parameters for substrate concentration
    }


    def __init__(self, num_substrates, num_receptors, parameters=None):
        """ initialize a receptor library by setting the number of receptors,
        the number of substrates it can respond to, and optional additional
        parameters in the parameter dictionary """
        super(LibrarySparseBase, self).__init__(num_substrates, num_receptors,
                                                parameters)

        # apply the parameters to the object
        if self.parameters['concentration_parameters'] is None:
            self.concentrations = self.parameters['concentration_vector']
        else:
            self.set_concentrations(**self.parameters['concentration_parameters'])


    @classmethod
    def get_random_arguments(cls, **kwargs):
        """ create random args for creating test instances """
        args = super(LibrarySparseBase, cls).get_random_arguments(**kwargs)
        if kwargs.get('homogeneous_mixture', False):
            ds = np.full(args['num_substrates'], np.random.random() + 0.5)
        else:
            ds = np.random.random(args['num_substrates']) + 0.5
        args['parameters'] = {'concentration_vector': ds}
        return args
    
    
    @property
    def concentrations(self):
        """ return the concentrations vector """
        return self._ds
    
    @concentrations.setter
    def concentrations(self, ds):
        """ sets the substrate concentrations """
        if ds is None:
            # initialize with default values, but don't save the parameters
            self._ds = np.ones(self.Ns)
            
        else:
            if len(ds) != self.Ns:
                raise ValueError('Length of the concentration vector must '
                                 'match the number of substrates.')
            if any(ds < 0):
                raise ValueError('Concentration vector must not contain '
                                 'negative entries.')
            self._ds = np.asarray(ds)
            
            # save the values, since they were set explicitly 
            self.parameters['concentration_vector'] = self._ds
    
    
    def get_concentration_distribution(self, i):
        """ returns the concentration distribution for component i """
        return stats.expon(scale=self.concentrations[i])

    
    def get_concentration_means(self):
        """ returns the mean concentration with which each substrate is
        expected """
        return self.substrate_probabilities * self.concentrations
    
    
    @property
    def is_homogeneous(self):
        """ returns True if the mixture is homogeneous """
        h_i = self.commonness
        d_i = self.concentrations
        return np.allclose(h_i, h_i[0]) and np.allclose(d_i, d_i[0])
            
    
    def set_concentrations(self, scheme, total_concentration, **kwargs):
        """ picks a commonness vector according to the supplied parameters:
        `total_concentration` sets the total concentration to expect for the
            mixture on average.
        """
        mean_concentration = total_concentration / self.Ns
        
        if scheme == 'const':
            # all substrates are equally likely
            c_means = np.full(self.Ns, mean_concentration)
                
        elif scheme == 'random_uniform':
            # draw the mean probabilities from a uniform distribution
            c_means = np.random.uniform(0, 2*mean_concentration, self.Ns)
            # make sure that the mean concentration is correct
            c_means *= total_concentration/c_means.sum()
            
        else:
            raise ValueError('Unknown commonness scheme `%s`' % scheme)
        
        # set the concentration
        self.concentrations = c_means
                
        # we additionally store the parameters that were used for this function
        c_params = {'scheme': scheme, 'total_concentration': total_concentration}
        c_params.update(kwargs)
        self.parameters['concentration_parameters'] = c_params  

