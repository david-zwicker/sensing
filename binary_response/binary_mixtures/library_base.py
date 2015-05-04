'''
Created on Apr 1, 2015

@author: David Zwicker <dzwicker@seas.harvard.edu>
'''

from __future__ import division

import numpy as np

from ..library_base import LibraryBase



class LibraryBinaryBase(LibraryBase):
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
        super(LibraryBinaryBase, self).__init__(num_substrates, num_receptors,
                                                parameters)

        # apply the parameters to the object
        if self.parameters['commonness_parameters'] is None:
            self.commonness = self.parameters['commonness_vector']
        else:
            self.set_commonness(**self.parameters['commonness_parameters'])


    @classmethod
    def get_random_arguments(cls, **kwargs):
        """ create random arguments for creating test instances """
        args = super(LibraryBinaryBase, cls).get_random_arguments(**kwargs)
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
            self._hs = np.zeros(self.Ns)
            self._ps = self._hs + 0.5
            
        else:
            if len(hs) != self.Ns:
                raise ValueError('Length of the commonness vector must match the '
                                 'number of substrates.')
            self._hs = np.asarray(hs)
            self._ps = 1/(1 + np.exp(-self._hs))
            
            # save the values, since they were set explicitly 
            self.parameters['commonness_vector'] = self._hs
    
    
    @property
    def substrate_probability(self):
        """ return the probability of finding each substrate """
        return self._ps
    
    @substrate_probability.setter
    def substrate_probability(self, ps):
        """ sets the substrate probability and the associated commonness """
        if len(ps) != self.Ns:
            raise ValueError('Length of the probability vector must match the '
                             'number of substrates.')
        ps = np.asarray(ps)
        if np.any(ps < 0) or np.any(ps > 1):
            raise ValueError('All probabilities must be within [0, 1]')
        
        with np.errstate(all='ignore'):
            self._hs = np.log(ps) - np.log1p(-ps)
        self._ps = ps
        
        # save the values, since they were set explicitly 
        self.parameters['commonness_vector'] = self._hs
            
    
    @property
    def is_homogeneous(self):
        """ returns True if the mixture is homogeneous """
        h_i = self.commonness
        return np.allclose(h_i, h_i[0])
            
            
    def mixture_size_distribution(self):
        """ calculates the probabilities of finding a mixture with a given
        number of components. Returns an array of length Ns + 1 of probabilities
        for finding mixtures with the number of components given by the index
        into the array """
        res = np.zeros(self.Ns + 1)
        res[0] = 1
        # iterate over each substrate and consider its individual probability
        for k, p in enumerate(self.substrate_probability, 1):
            res[k] = res[k-1]*p
            res[1:k] = (1 - p)*res[1:k] + res[:k-1]*p
            res[0] = (1 - p)*res[0]
            
        return res
            
            
    def mixture_size_statistics(self):
        """ calculates the mean and the standard deviation of the number of
        components in mixtures """
        prob_s = self.substrate_probability
        l_mean = np.sum(prob_s)
        l_var = np.sum(prob_s/(1 + np.exp(self._hs)))
        
        return l_mean, np.sqrt(l_var)
    
    
    def set_commonness(self, scheme='const', mean_mixture_size=1, **kwargs):
        """ picks a commonness vector according to the supplied parameters:
        `mean_mixture_size` is determines the mean number of components in each
        mixture. The value of the commonness vector are furthermore influenced  
        by the `scheme`, which can be any of the following:
            `const`: all substrates are equally probable
            `single`: the first substrate has a different probability, which can
                either be specified directly by supplying the parameter `p1` or
                the `ratio` between p1 and the probabilities of the other
                substrates can be specified.
            `geometric`: the probability of substrates decreases by a factor of
                `alpha` from each substrate to the next.
            `linear`: the probability of substrates decreases linearly.
            `random_uniform`: the probability of substrates is chosen from a
                uniform distribution with given mean and maximal variance.
        """
        if scheme == 'const':
            # all substrates are equally likely
            ps = np.full(self.Ns, mean_mixture_size/self.Ns)
        
        elif scheme == 'single':
            # the first substrate has a different commonness than the others
            ps = np.empty(self.Ns)
            if 'p1' in kwargs:
                # use the given probability for the first substrate
                ps[0] = kwargs['p1']
                ps[1:] = (mean_mixture_size - ps[0]) / (self.Ns - 1)
                 
            elif 'p_ratio' in kwargs:
                # use the given ratio between the first and the other substrates
                ratio = kwargs['p_ratio']
                denom = self.Ns + ratio - 1
                ps[0] = mean_mixture_size * ratio / denom
                ps[1:] = mean_mixture_size / denom
                
            else:
                raise ValueError('Either `p1` or `p_ratio` must be given')

            
        elif scheme == 'geometric':
            # substrates have geometrically decreasing commonness 
            alpha = kwargs.pop('alpha', 0.9)

            if alpha == 1:
                p0 = mean_mixture_size/self.Ns
            else:
                p0 = mean_mixture_size * (1 - alpha) / (1 - alpha**self.Ns)
                
            if p0 > 1:
                raise ValueError('It is not possible to choose commonness '
                                 'parameters such that the mean mixture size '
                                 'of %d components is %g for alpha=%g'
                                 % (self.Ns, mean_mixture_size, alpha))
                
            ps = p0 * alpha**np.arange(self.Ns)
            
        elif scheme == 'linear':
            # substrates have a linear decreasing probability
            if mean_mixture_size <= 0.5*self.Ns:
                a, b = 0, 2*mean_mixture_size/self.Ns
            else:
                a, b = 2*mean_mixture_size/self.Ns - 1, 1
                
            ps = np.linspace(a, b, self.Ns)
            
        elif scheme == 'random_uniform':
            # substrates have random probability chosen from a uniform dist.
            if mean_mixture_size <= 0.5*self.Ns:
                a, b = 0, 2*mean_mixture_size/self.Ns
            else:
                a, b = 2*mean_mixture_size/self.Ns - 1, 1
                
            # choose random probabilities
            ps = np.random.uniform(a, b, size=self.Ns)
            ps_mean = ps.sum()
            
            # correct the probabilities to ensure the mean
            if ps_mean < mean_mixture_size:
                # increase ps to match mean 
                ps_c = 1 - ps #< consider the compliment
                ps_c *= (self.Ns - mean_mixture_size)/(self.Ns - ps_mean)
                ps = 1 - ps_c
            else:
                # decrease ps to match mean
                ps *= mean_mixture_size/ps_mean
            
        else:
            raise ValueError('Unknown commonness scheme `%s`' % scheme)
        
        # set the probability which also calculates the commonness and saves
        # the values in the parameters dictionary
        self.substrate_probability = ps
        
        # we additionally store the parameters that were used for this function
        c_params = {'scheme': scheme, 'mean_mixture_size': mean_mixture_size}
        c_params.update(kwargs)
        self.parameters['commonness_parameters'] = c_params  

