'''
Created on Apr 1, 2015

@author: David Zwicker <dzwicker@seas.harvard.edu>
'''

from __future__ import division

import multiprocessing as mp

import numpy as np



class LibraryBase(object):
    """ represents a single receptor library. This is a base class that provides
    general functionality and parameter management.
    
    For instance, the class provides a framework for calculating ensemble
    averages, where each time new commonness vectors are chosen randomly
    according to the parameters of the last call to `set_commonness`.  
    """

    # default parameters that are used to initialize a class if not overwritten
    parameters_default = {
        'random_seed': None,           #< seed for the random number generator
        'ensemble_average_num': 32,    #< repetitions for ensemble average
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
        if parameters is not None:
            self.parameters.update(parameters)
        
        np.random.seed(self.parameters['random_seed'])


    @property
    def init_arguments(self):
        """ return the parameters of the model that can be used to reconstruct
        it by calling the __init__ method with these arguments """
        return {'num_substrates': self.Ns,
                'num_receptors': self.Nr,
                'parameters': self.parameters}


    @classmethod
    def get_random_arguments(cls, **kwargs):
        """ create random arguments for creating test instances """
        Ns = kwargs.get('Ns', np.random.randint(3, 6))
        Nr = kwargs.get('Nr', np.random.randint(2, 4))
        return [Ns, Nr]


    @classmethod
    def create_test_instance(cls, **kwargs):
        """ creates a test instance used for consistency tests """
        return cls(*cls.get_random_arguments(**kwargs))
  
            
    def ensemble_average(self, method, avg_num=None, multiprocessing=False, 
                         ret_all=False):
        """ calculate an ensemble average of the result of the `method` of
        multiple different receptor libraries """
        
        if avg_num is None:
            avg_num = self.parameters['ensemble_average_num']
        
        if multiprocessing:
            # run the calculations in multiple processes  
            arguments = (self.__class__, self.init_arguments, method)
            pool = mp.Pool()
            result = pool.map(_ReceptorLibrary_mp_calc, [arguments] * avg_num)
            
        else:
            # run the calculations in this process
            result = [getattr(self.__class__(**self.init_arguments), method)()
                      for _ in xrange(avg_num)]
    
        # collect the results and calculate the statistics
        result = np.array(result)
        if ret_all:
            return result
        else:
            return result.mean(axis=0), result.std(axis=0)

        

def _ReceptorLibrary_mp_calc(args):
    """ helper function for multiprocessing """
    obj = args[0](**args[1])
    return getattr(obj, args[2])()

    
    