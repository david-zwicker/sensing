'''
Created on Mar 31, 2015

@author: David Zwicker <dzwicker@seas.harvard.edu>
'''

from __future__ import division

import numpy as np

from .library_base import LibraryBinaryBase



LN2 = np.log(2)



class LibraryBinaryUniform(LibraryBinaryBase):
    """ represents a single receptor library with random entries. The only
    parameters that characterizes this library is the density of entries. """


    def __init__(self, num_substrates, num_receptors, density=1,
                 parameters=None):
        """ initialize the receptor library by setting the number of receptors,
        the number of substrates it can respond to, and the fraction `density`
        of substrates a single receptor responds to """
        super(LibraryBinaryUniform, self).__init__(num_substrates,
                                                   num_receptors, parameters)
        self.density = density


    @property
    def repr_params(self):
        """ return the important parameters that are shown in __repr__ """
        params = super(LibraryBinaryUniform, self).repr_params
        params.append('xi=%g' % self.density)
        return params


    @property
    def init_arguments(self):
        """ return the parameters of the model that can be used to reconstruct
        it by calling the __init__ method with these arguments """
        args = super(LibraryBinaryUniform, self).init_arguments
        args['density'] = self.density
        return args


    @classmethod
    def get_random_arguments(cls, **kwargs):
        """ create random arguments for creating test instances """
        args = super(LibraryBinaryUniform, cls).get_random_arguments(**kwargs)
        args['density'] = kwargs.get('density', np.random.random())
        return args

    
    def receptor_activity(self, ret_correlations=False, approx_prob=False,
                          clip=True):
        """ return the probability with which a single receptor is activated 
        by typical mixtures """
        q_n, q_nm = self.receptor_crosstalk(ret_receptor_activity=True,
                                            approx_prob=approx_prob)
        
        r_n = q_n
        r_nm = q_n**2 + q_nm
        
        if clip:
            r_n = np.clip(r_n, 0, 1)
            r_nm = np.clip(r_nm, 0, 1)
        
        if ret_correlations:
            return r_n, r_nm
        else:
            return r_n
        
        
    def receptor_crosstalk(self, ret_receptor_activity=False, approx_prob=False):
        """ calculates the average activity of the receptor as a response to 
        single ligands. """

        p_i = self.substrate_probabilities
        
        # get probability q_n and q_nm that receptors are activated 
        if approx_prob:
            # use approximate formulas for calculating the probabilities
            q_n = self.density * p_i.sum()
            q_nm = self.density**2 * p_i.sum()
            
            # clip the result to [0, 1]
            q_n = np.clip(q_n, 0, 1)
            q_nm = np.clip(q_nm, 0, 1)

        else:
            # use better formulas for calculating the probabilities 
            q_n = 1 - np.prod(1 - self.density * p_i)
            q_nm = 1 - np.prod(1 - self.density**2 * p_i)
                
        if ret_receptor_activity:
            return q_n, q_nm
        else:
            return q_nm

        
    def mutual_information(self, approx_prob=False, use_polynom=False):
        """ return a theoretical estimate of the mutual information between
        input and output.
            `approx_prob` determines whether a linear approximation should be
                used to calculate the probabilities that receptors are active
            `use_polynom` determines whether a polynomial approximation for the
                mutual information should be used
        """
        if use_polynom:
            # use the expansion of the mutual information around the optimal
            # point to calculate an approximation of the mututal information
            
            # determine the probabilities of receptor activations        
            q_n, q_nm = self.receptor_crosstalk(ret_receptor_activity=True,
                                                approx_prob=approx_prob)
    
            # calculate mutual information from this
            MI = self._estimate_mutual_information_from_q_stats(
                                                q_n, q_nm, use_polynom=True)

        else:
            # calculate the MI assuming that receptors are independent.
            # This expression assumes that each receptor provides a fractional 
            # information H_r/N_s. Some of the information will be overlapping
            # and the resulting MI is thus smaller than the naive estimate:
            #     MI < N_r * H_r

            # determine the probabilities of receptor activation  
            q_n = self.receptor_activity(approx_prob=approx_prob)
    
            # calculate mutual information from this, ignoring crosstalk
            MI = self._estimate_mutual_information_from_q_stats(
                                                    q_n, 0, use_polynom=False)

            # estimate the effect of crosstalk by calculating the expected
            # overlap between independent receptors  
            H_r = MI / self.Nr
            MI = self.Ns - self.Ns*(1 - H_r/self.Ns)**self.Nr
        
        # limit the MI to the mixture entropy
        return min(MI, self.mixture_entropy())
        
        
    def density_optimal(self, assume_homogeneous=False):
        """ return the estimated optimal activity fraction for the simple case
        where all h are the same. The estimate relies on an approximation that
        all receptors are independent and is thus independent of the number of 
        receptors. The estimate is thus only good in the limit of low Nr.
        
        If `assume_homogeneous` is True, the calculation is also done in the
            case of heterogeneous mixtures, where the probability of the
            homogeneous system with the same average number of substrates is
            used instead.
        """
        if not assume_homogeneous and len(np.unique(self.commonness)) > 1:
            # mixture is heterogeneous
            raise RuntimeError('The estimate only works for homogeneous '
                               'mixtures so far.')
                
        # mean probability of finding a specific substrate in a mixture
        p0 = self.substrate_probabilities.mean()
            
        # calculate the fraction for the homogeneous case
        return (1 - 2**(-1/self.Ns))/p0
    
    
    def get_optimal_library(self):
        """ returns an estimate for the optimal parameters for the random
        interaction matrices """
        return {'density': self.density_optimal(assume_homogeneous=True)}
        
        