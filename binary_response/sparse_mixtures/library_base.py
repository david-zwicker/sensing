'''
Created on Apr 1, 2015

@author: David Zwicker <dzwicker@seas.harvard.edu>
'''

from __future__ import division

import numpy as np
from scipy import stats, special

from ..binary_mixtures.library_base import LibraryBinaryBase



@np.vectorize
def _estimate_qn_from_en_approx(en_mean, en_var):
    """ estimates probability q_n that a receptor is activated by a mixture
    based on the statistics of the excitations s_n using an approximation """
    if en_var == 0:
        q_n = np.float(en_mean > 1)
    else:                
        delta = (en_mean - 1) / np.sqrt(en_var)
        q_n = 0.5 + delta / np.sqrt(2*np.pi)

    return q_n



@np.vectorize
def _estimate_qn_from_en_lognorm(en_mean, en_var):
    """ estimates probability q_n that a receptor is activated by a mixture
    based on the statistics of the excitations s_n assuming an underlying
    log-normal distribution for s_n """
    if en_var == 0:
        q_n = np.float(en_mean > 1)
    elif en_mean == 0:
        q_n = 0
    else:
        en_cv2 = en_var / en_mean**2
        enum = np.log(np.sqrt(1 + en_cv2) / en_mean)
        denom = np.sqrt(2*np.log(1 + en_cv2))
        q_n = 0.5 * special.erfc(enum/denom)
        
    return q_n



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

        initialize_state = self.parameters['initialize_state'] 
        if initialize_state is None:
            # do not initialize with anything
            self.concentrations = None
            
        elif initialize_state == 'exact':
            # initialize the state using saved parameters
            self.concentrations = self.parameters['concentration_vector']
            
        elif initialize_state == 'ensemble':
            # initialize the state using the ensemble parameters
            self.set_concentrations(**self.parameters['concentration_parameters'])
            
        elif initialize_state == 'auto':
            # use exact values if saved or ensemble properties otherwise
            if self.parameters['concentration_parameters'] is None:
                self.concentrations = self.parameters['concentration_vector']
            else:
                self.set_concentrations(**self.parameters['concentration_parameters'])
        
        else:
            raise ValueError('Unknown initialization protocol `%s`' % 
                             initialize_state)


    @property
    def repr_params(self):
        """ return the important parameters that are shown in __repr__ """
        params = super(LibrarySparseBase, self).repr_params
        params.append('<d>=%g' % self.concentrations.mean())
        return params


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
    
    
    @property
    def concentration_means(self):
        """ return the mean concentration at which each substrate is expected
        on average """
        return self.substrate_probabilities * self.concentrations
    
    
    def get_concentration_distribution(self, i):
        """ returns the concentration distribution for component i """
        return stats.expon(scale=self.concentrations[i])

    
    def concentration_statistics(self):
        """ returns statistics for each individual substrate """
        if self.correlated_mixture:
            raise NotImplementedError('Not implemented for correlated mixtures')

        pi = self.substrate_probabilities
        di = self.concentrations
        c_means = pi * di
        c_vars = pi*(2 - pi) * di**2
        
        # return the results in a dictionary to be able to extend it later
        return {'mean': c_means, 'std': np.sqrt(c_vars), 'var': c_vars}

    
    @property
    def is_homogeneous(self):
        """ returns True if the mixture is homogeneous """
        h_i = self.commonness
        d_i = self.concentrations
        return np.allclose(h_i, h_i[0]) and np.allclose(d_i, d_i[0])
            
    
    def set_concentrations(self, scheme, mean_concentration, **kwargs):
        """ picks a commonness vector according to the supplied parameters:
        `total_concentration` sets the total concentration to expect for the
            mixture on average.
        """
        
        if scheme == 'const':
            # all substrates are equally likely
            c_means = np.full(self.Ns, mean_concentration)
                
        elif scheme == 'random_uniform':
            # draw the mean probabilities from a uniform distribution
            c_means = np.random.uniform(0, 2*mean_concentration, self.Ns)
            
        else:
            raise ValueError('Unknown commonness scheme `%s`' % scheme)

        # make sure that the mean concentration is correct
        c_means *= mean_concentration / c_means.mean()
        
        # set the concentration
        self.concentrations = c_means
                
        # we additionally store the parameters that were used for this function
        c_params = {'scheme': scheme, 'mean_concentration': mean_concentration}
        c_params.update(kwargs)
        self.parameters['concentration_parameters'] = c_params
        
        
    def _estimate_qn_from_en(self, en_mean, en_var, approx_prob=False):  
        """ estimates probability q_n that a receptor is activated by a mixture
        based on the statistics of the excitations en """

        if approx_prob:
            q_n = _estimate_qn_from_en_approx(en_mean, en_var)

        else:
            # estimate from a log-normal distribution
            q_n = _estimate_qn_from_en_lognorm(en_mean, en_var)
            
        return q_n
   
    
    def _estimate_qnm_from_en(self, en_var, enm_covar):
        """ estimates crosstalk q_nm based on the statistics of the excitations
        en """
        en_std = np.sqrt(en_var)
        
        # calculate the correlation coefficient 
        with np.errstate(divide='ignore', invalid='ignore'):
            rho = np.divide(enm_covar, np.outer(en_std, en_std))

        # Replace values that are nan with zero. This might not be exact,
        # but only occurs in corner cases that are not interesting to us  
        rho[np.isnan(rho)] = 0
            
        # estimate the crosstalk
        q_nm = rho / (2*np.pi)
            
        return q_nm
    
        