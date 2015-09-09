'''
Created on Apr 1, 2015

@author: David Zwicker <dzwicker@seas.harvard.edu>
'''

from __future__ import division

import numpy as np
from scipy import stats

from ..library_base import LibraryBase



class LibraryExponentialBase(LibraryBase):
    """ represents a single receptor library. This is a base class that provides
    general functionality and parameter management.
    
    For instance, the class provides a framework for calculating ensemble
    averages, where each time new concentration vectors are chosen randomly
    according to the parameters of the last call to `choose_concentrations`.  
    """

    # default parameters that are used to initialize a class if not overwritten
    parameters_default = {
        'concentrations_vector': None,     #< chosen substrate concentration
        'concentrations_parameters': None, #< parameters for substrate concentration
    }


    def __init__(self, num_substrates, num_receptors, parameters=None):
        """ initialize a receptor library by setting the number of receptors,
        the number of substrates it can respond to, and optional additional
        parameters in the parameter dictionary """
        super(LibraryExponentialBase, self).__init__(num_substrates,
                                                     num_receptors,
                                                     parameters)

        initialize_state = self.parameters['initialize_state'] 
        if initialize_state is None:
            # do not initialize with anything
            self.concentrations = None
            
        elif initialize_state == 'exact':
            # initialize the state using saved parameters
            self.concentrations = self.parameters['concentrations_vector']
            
        elif initialize_state == 'ensemble':
            # initialize the state using the ensemble parameters
            self.choose_concentrations(**self.parameters['concentrations_parameters'])
            
        elif initialize_state == 'auto':
            # use exact values if saved or ensemble properties otherwise
            if self.parameters['concentrations_parameters'] is None:
                self.concentrations = self.parameters['concentrations_vector']
            else:
                self.choose_concentrations(**self.parameters['concentrations_parameters'])
                
        else:
            raise ValueError('Unknown initialization protocol `%s`' % 
                             initialize_state)


    @property
    def repr_params(self):
        """ return the important parameters that are shown in __repr__ """
        params = super(LibraryExponentialBase, self).repr_params
        params.append('<c_i>=%g' % self.concentration_means)
        return params


    @classmethod
    def get_random_arguments(cls, homogeneous_mixture=False,  **kwargs):
        """ create random arguments for creating test instances """
        args = super(LibraryExponentialBase, cls).get_random_arguments(**kwargs)
        
        if homogeneous_mixture:
            p_i = np.full(args['num_substrates'], np.random.random() + 0.5)
        else:
            p_i = np.random.random(args['num_substrates']) + 0.5
            
        args['parameters'] = {'commonness_vector': p_i}
        return args


    @property
    def concentrations(self):
        """ return the concentrations vector """
        return self._pi
    
    @concentrations.setter
    def concentrations(self, p_i):
        """ sets the concentrations and the associated substrate probability """
        if p_i is None:
            # initialize with default values, but don't save the parameters
            self._pi = np.ones(self.Ns)
            
        else:
            if len(p_i) != self.Ns:
                raise ValueError('Length of the concentrations vector must match '
                                 'the number of substrates.')
            if any(p_i < 0):
                raise ValueError('Concentration vector must only contain '
                                 'non-negative entries.')
            self._pi = np.asarray(p_i)
            
            # save the values, since they were set explicitly 
            self.parameters['concentrations_vector'] = self._pi

    
    @property
    def concentration_means(self):
        """ returns the mean concentrations with which each substrate is
        expected """
        return self.concentrations
    
    
    def get_concentration_distribution(self, i):
        """ returns the concentrations distribution for component i """
        return stats.expon(scale=self.concentrations[i])

    
    def concentration_statistics(self):
        """ returns statistics for each individual substrate """
        p_i = self.concentrations
        c_means = p_i
        c_vars = p_i**2
        # return the results in a dictionary to be able to extend it later
        return {'mean': c_means, 'std': np.sqrt(c_vars), 'var': c_vars}
    
    
    @property
    def is_homogeneous_mixture(self):
        """ returns True if the mixture is homogeneous """
        p_i = self.concentrations
        return np.allclose(p_i, p_i.mean())
            
    
    def choose_concentrations(self, scheme, total_concentration, **kwargs):
        """ picks a concentration vector according to the supplied parameters:
        `total_concentration` sets the total concentration to expect for the
            mixture on average.
        """
        mean_concentration = total_concentration / self.Ns
        
        if scheme == 'const':
            # all substrates are equally likely
            p_i = np.full(self.Ns, mean_concentration)
                
        elif scheme == 'random_uniform':
            # draw the mean probabilities from a uniform distribution
            p_i = np.random.uniform(0, 2*mean_concentration, self.Ns)
            # make sure that the mean concentration is correct
            p_i *= total_concentration / p_i.sum()
            
        else:
            raise ValueError('Unknown concentration scheme `%s`' % scheme)
        
        self.concentrations = p_i
        
        # we additionally store the parameters that were used for this function
        c_params = {'scheme': scheme, 'total_concentration': total_concentration}
        c_params.update(kwargs)
        self.parameters['concentrations_parameters'] = c_params  
   

