'''
Created on Apr 1, 2015

@author: David Zwicker <dzwicker@seas.harvard.edu>
'''

from __future__ import division

import multiprocessing as mp

import numpy as np


LN2 = np.log(2)


class LibraryBase(object):
    """ represents a single receptor library. This is a base class that provides
    general functionality and parameter management.
    
    For instance, the class provides a framework for calculating ensemble
    averages, where each time new commonness vectors are chosen randomly
    according to the parameters of the last call to `set_commonness`.  
    """

    # default parameters that are used to initialize a class if not overwritten
    parameters_default = {
        'initialize_state': 'auto',      #< how to initialize the state
        'ensemble_average_num': 32,      #< repetitions for ensemble average
        'multiprocessing_cores': 'auto', #< number of cores to use
    }


    def __init__(self, num_substrates, num_receptors, parameters=None):
        """ initialize a receptor library by setting the number of receptors,
        the number of substrates it can respond to, and optional additional
        parameters in the parameter dictionary """
        self.Ns = num_substrates
        self.Nr = num_receptors

        # initialize parameters with default ones from all parent classes
        self.parameters = {}
        for cls in reversed(self.__class__.__mro__):
            if hasattr(cls, 'parameters_default'):
                self.parameters.update(cls.parameters_default)
        # update parameters with the supplied ones
        if parameters is not None:
            self.parameters.update(parameters)


    @property
    def repr_params(self):
        """ return the important parameters that are shown in __repr__ """
        return ['Ns=%d' % self.Ns, 'Nr=%d' % self.Nr]


    def __repr__(self):
        params = ', '.join(self.repr_params)
        return '%s(%s)' % (self.__class__.__name__, params)


    @property
    def init_arguments(self):
        """ return the parameters of the model that can be used to reconstruct
        it by calling the __init__ method with these arguments """
        return {'num_substrates': self.Ns,
                'num_receptors': self.Nr,
                'parameters': self.parameters}

    
    def copy(self):
        """ returns a copy of the current object """
        return self.__class__(**self.init_arguments)


    @classmethod
    def from_other(cls, other, **kwargs):
        """ creates an instance of this class by using parameters from another
        instance """
        # create object with parameters from other object
        init_arguments = other.init_arguments
        init_arguments.update(kwargs)
        return cls(**init_arguments)


    @classmethod
    def get_random_arguments(cls, **kwargs):
        """ create random arguments for creating test instances """
        return {'num_substrates': kwargs.get('Ns', np.random.randint(3, 6)),
                'num_receptors': kwargs.get('Nr', np.random.randint(2, 4))}


    @classmethod
    def create_test_instance(cls, **kwargs):
        """ creates a test instance used for consistency tests """
        return cls(**cls.get_random_arguments(**kwargs))
  
            
    def get_number_of_cores(self):
        """ returns the number of cores to use in multiprocessing """
        multiprocessing_cores = self.parameters['multiprocessing_cores']
        if multiprocessing_cores == 'auto':
            return mp.cpu_count()
        else:
            return multiprocessing_cores
            
            
    def ensemble_average(self, method, avg_num=None, multiprocessing=False, 
                         ret_all=False, args=None):
        """ calculate an ensemble average of the result of the `method` of
        multiple different receptor libraries """
        
        if avg_num is None:
            avg_num = self.parameters['ensemble_average_num']
        if args is None:
            args = {}
        
        if multiprocessing and avg_num > 1:
            
            init_arguments = self.init_arguments
            init_arguments['parameters']['initialize_state'] = 'ensemble'
            
            # run the calculations in multiple processes
            arguments = (self.__class__, init_arguments, method, args)
            pool = mp.Pool(processes=self.get_number_of_cores())
            result = pool.map(_ensemble_average_job, [arguments] * avg_num)
            
            # Apparently, multiprocessing sometimes opens too many files if
            # processes are launched to quickly and the garbage collector cannot
            # keep up. We thus explicitly terminate the pool here.
            pool.terminate()
            
        else:
            # run the calculations in this process
            result = [getattr(self.__class__(**self.init_arguments), method)(**args)
                      for _ in range(avg_num)]
    
        # collect the results and calculate the statistics
        if ret_all:
            return result
        else:
            result = np.array(result)
            return result.mean(axis=0), result.std(axis=0)
        
        
    def _estimate_mutual_information_from_q_values(self, q_n, q_nm,
                                                   use_polynom=False):
        """ estimate the mutual information from given probabilities """
        # calculate the approximate mutual information from data
        MI = self.Nr
        MI -= 0.5/LN2 * np.sum((2*q_n - 1)**2)
        MI -= 8/LN2 * np.sum(np.triu(q_nm, 1)**2)
            
        return MI
    
        
    def _estimate_mutual_information_from_q_stats(
            self, q_n, q_nm, q_n_var=0, q_nm_var=0,
            use_polynom=True, ret_var=False):
        """ estimate the mutual information from given probabilities """
        Nr = self.Nr
        
        if use_polynom:
            # use the quadratic approximation of the mutual information
            MI = Nr - 0.5/LN2 * Nr * ((2*q_n - 1)**2 + 4*q_n_var)
            
        else:
            # calculate the information assuming receptors are independent            
            MI = -Nr * (q_n*np.log2(q_n) + (1 - q_n)*np.log2(1 - q_n))
        
        # add the effect of crosstalk
        MI -= 4/LN2 * Nr*(Nr - 1) * (q_nm**2 + q_nm_var)
            
        if ret_var:
            # also estimate the variance of the mutual information
            MI_var = (
                4 * Nr**2 * q_n_var * ((2*q_n - 1)**2 + 2*q_n_var)
                + 32 * (Nr*(Nr - 1))**2 * q_nm_var * (2*q_nm**2 + q_nm_var)
            ) / LN2**2
            return MI, MI_var
        else:
            return MI
    
        
    def _estimate_mutual_information_from_r_values(self, r_n, r_nm):
        """ estimate the mutual information from given probabilities """
        # calculate the crosstalk
        q_nm = r_nm - np.outer(r_n, r_n)
        return self._estimate_mutual_information_from_q_values(r_n, q_nm)
      
        
    def _estimate_mutual_information_from_r_stats(self, r_n, r_nm, r_n_var=0,
                                                  r_nm_var=0, ret_var=False):
        """ estimate the mutual information from given probabilities """
        if r_nm_var != 0:
            raise NotImplementedError('Correlation calculations are not tested.')
        # calculate the crosstalk
        q_nm = r_nm - r_n**2 - r_n_var
        q_nm_var = r_nm_var + 4*r_n**2*r_n_var + 2*r_n_var**2
        return self._estimate_mutual_information_from_q_stats(
                                  r_n, q_nm, r_n_var, q_nm_var, ret_var=ret_var)
      
    

def _ensemble_average_job(args):
    """ helper function for calculating ensemble averages using
    multiprocessing """
    # We have to initialize the random number generator for each process
    # because we would have the same random sequence for all processes
    # otherwise.
    np.random.seed()
    
    # create the object ...
    obj = args[0](**args[1])
    # ... and evaluate the requested method
    if len(args) > 2: 
        return getattr(obj, args[2])(**args[3])
    else:
        return getattr(obj, args[2])()

    
    