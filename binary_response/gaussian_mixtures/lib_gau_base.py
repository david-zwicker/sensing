'''
Created on Apr 1, 2015

@author: David Zwicker <dzwicker@seas.harvard.edu>
'''

from __future__ import division

import logging

import numpy as np
from scipy import stats

from ..library_base import LibraryBase
from utils.misc import is_pos_semidef



class LibraryGaussianBase(LibraryBase):
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
        'covariance_matrix': None,         #< chosen substrate covariance matrix
        'covariance_parameters': None,     #< parameters for substrate covariance
    }


    def __init__(self, num_substrates, num_receptors, parameters=None):
        """ initialize a receptor library by setting the number of receptors,
        the number of substrates it can respond to, and optional additional
        parameters in the parameter dictionary """
        super(LibraryGaussianBase, self).__init__(num_substrates,
                                                  num_receptors,
                                                  parameters)

        # determine how to initialize the variables
        init_state = self.parameters['initialize_state']
        
        # determine how to initialize the concentrations
        init_concentrations = init_state.get('concentrations',
                                             init_state['default'])
        if init_concentrations  == 'auto':
            if self.parameters['concentrations_parameters'] is None:
                init_concentrations = 'exact'
            else:
                init_concentrations = 'ensemble'

        # initialize the concentrations with the chosen method            
        if init_concentrations is None:
            self.concentrations = None
            
        elif init_concentrations  == 'exact':
            logging.debug('Initialize with given concentrations')
            self.concentrations = self.parameters['concentrations_vector']
            
        elif init_concentrations == 'ensemble':
            conc_params = self.parameters['concentrations_parameters']
            if conc_params:
                logging.debug('Choose concentrations from given parameters')
                self.choose_concentrations(**conc_params)
            else:
                logging.warn('Requested to set concentrations from parameters, '
                             'but parameters were not supplied.')
                self.concentrations = None
                    
        else:
            raise ValueError('Unknown initialization protocol `%s`' % 
                             init_concentrations)
        
        # determine how to initialize the covariance matrix
        init_covariance = init_state.get('covariance', init_state['default'])
        if init_covariance  == 'auto':
            if self.parameters['covariance_parameters'] is None:
                init_covariance = 'exact'
            else:
                init_covariance = 'ensemble'
                
        # initialize the covariance with the chosen method            
        if init_covariance is None:
            self.covariance = None
            
        elif init_covariance == 'exact':
            logging.debug('Initialize with given covariance')
            self.covariance = self.parameters['covariance_matrix']
            
        elif init_covariance == 'ensemble':
            covariance_parameters = self.parameters['covariance_parameters']
            if covariance_parameters:
                logging.debug('Choose covariance matrix from given parameters')
                self.choose_covariance(**covariance_parameters)
            else:
                logging.warn('Requested to set covariance from parameters, '
                             'but parameters were not supplied.')
                self.covariance = None
        else:
            raise ValueError('Unknown initialization protocol `%s`' % 
                             init_covariance)
            
        # prepare indices that we need
        eye = np.eye(self.Ns).astype(np.bool)
        self._diag = np.nonzero(eye)
        self._offdiag = np.nonzero(~eye)


    @property
    def repr_params(self):
        """ return the important parameters that are shown in __repr__ """
        params = super(LibraryGaussianBase, self).repr_params
        params.append('<c_i>=%g' % self.concentration_means)
        return params


    @classmethod
    def get_random_arguments(cls, homogeneous_mixture=False, 
                             mixture_correlated=False, **kwargs):
        """ create random arguments for creating test instances """
        args = super(LibraryGaussianBase, cls).get_random_arguments(**kwargs)
        Ns = args['num_substrates']
        
        if homogeneous_mixture:
            p_i = np.full(args['num_substrates'], np.random.random() + 0.5)
        else:
            p_i = np.random.random(args['num_substrates']) + 0.5
            
        if mixture_correlated:
            p_ij = np.random.normal(size=(Ns, Ns))
            np.fill_diagonal(p_ij, 0)
            # the matrix will be symmetrize when it is set on the instance 
        else:
            p_ij = np.zeros((Ns, Ns))
            
        args['parameters'] = {'concenrations_vector': p_i,
                              'covariance_matrix': p_ij}
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
        c_vars = np.diag(self.covariance)
        # return the results in a dictionary to be able to extend it later
        return {'mean': c_means, 'std': np.sqrt(c_vars), 'var': c_vars,
                'cov': self.covariance}
    
    
    @property
    def is_homogeneous_mixture(self):
        """ returns True if the mixture is homogeneous """
        p_i = self.concentrations
        p_ij_diag = self.covariance[self._diag]
        p_ij_offdiag = self.covariance[self._offdiag]
        return (np.allclose(p_i, p_i.mean()) and
                np.allclose(p_ij_diag, p_ij_diag.mean()) and 
                np.allclose(p_ij_offdiag, 0))
            
    
    def choose_concentrations(self, scheme, total_concentration, **kwargs):
        """ picks a concentration vector according to the supplied parameters:
        `total_concentration` sets the total concentration to expect for the
            mixture on average.
        """
        mean_concentration = total_concentration / self.Ns
        
        if scheme == 'const':
            # all substrates are equally likely
            p_i = np.full(self.Ns, mean_concentration, np.double)
                
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
        

    @property
    def covariance(self):
        """ return the covariance matrix """
        return self._pij
    
    @covariance.setter
    def covariance(self, pij):
        """ sets the covariance matrix """
        if pij is None:
            # initialize with default values, but don't save the parameters
            self._pij = np.zeros((self.Ns, self.Ns))
            
        else:
            pij = np.asarray(pij)
            
            if pij.shape != (self.Ns, self.Ns):
                raise ValueError('Dimension of the covariance matrix must be '
                                 'Ns x Ns, where Ns is the number of '
                                 'substrates.')
                
            # symmetrize the matrix
            pij = np.tril(pij) + np.tril(pij, -1).T
        
            if not is_pos_semidef(pij):
                raise ValueError('Correlation matrix must be '
                                 'positive-semidefinite.')

            self._pij = pij 
        
            # save the values, since they were set explicitly
            self.parameters['covariance_matrix'] = self._pij
    
    
    @property 
    def is_correlated_mixture(self):
        """ returns True if the mixture has correlations, i.e. off-diagonal
        elements in the covariance matrix. """
        return np.count_nonzero(self.covariance[self._offdiag]) > 0        


    def choose_covariance(self, scheme, magnitude, **kwargs):
        """ picks a covariance matrix according to the supplied parameters:
        `magnitude` determines the magnitude of the covariance, which are
        drawn from the random distribution indicated by `scheme`: 
            `const`: all variances are equally to `magnitude` and the
                correlations are given by `correlation` * `magnitude`. The
                default value of the `correlation` is zero.
            `random_factors`: random scheme where a number of random Gaussian
                vectors are used to determine the covariance and the
                variances are drawn from a uniform distribution [0, 1]. The
                number of factors can be influenced by the `count` argument. 
        """
        shape = (self.Ns, self.Ns)

        cov_params = {'scheme': scheme, 'magnitude': magnitude}

        if scheme == 'const':
            # the covariance is a full matrix 
            correlation = kwargs.pop('correlation', 0)
            cov_params['correlation'] = correlation
            
            if correlation < 1/(1 - self.Ns) or correlation > 1:
                raise ValueError('The correlation parameter must be larger '
                                 'than 1/(1 - Ns) and smaller than 1.')
            
            # create matrix with 1 on diagonal and `correlation` otherwise
            p_ij = (np.full(shape, correlation, np.double)
                    + (1 - correlation) * np.eye(self.Ns))
            
        elif scheme == 'random_factors':
            # simple random scheme based on a number of factors
            correlation = kwargs.pop('correlation', 0)
            count = kwargs.pop('count', 5)
            cov_params['correlation'] = correlation
            cov_params['count'] = count

            W = correlation * np.random.randn(self.Ns, count)
            p_ij = np.dot(W, W.T) + np.diag(np.random.random(self.Ns))
            
            # variance normalization:
            #diag_sqrt = np.diag(1/np.sqrt(np.diag(p_ij))
            #p_ij = np.dot(np.dot(diag_sqrt, p_ij), diag_sqrt)
            
        else:
            raise ValueError('Unknown covariance scheme `%s`' % scheme)

        # scale the covariance matrix with the given magnitude
        self.covariance = magnitude * p_ij

        # we additionally store the parameters that were used for this function
        self.parameters['covariance_parameters'] = cov_params  
        
        # raise an error if keyword arguments have not been used
        if len(kwargs) > 0:
            raise ValueError('The following keyword arguments have not been '
                             'used: %s' % str(kwargs)) 
            
