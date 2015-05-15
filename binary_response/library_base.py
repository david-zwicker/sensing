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
    def get_random_arguments(cls, **kwargs):
        """ create random arguments for creating test instances """
        return {'num_substrates': kwargs.get('Ns', np.random.randint(3, 6)),
                'num_receptors': kwargs.get('Nr', np.random.randint(2, 4))}


    @classmethod
    def create_test_instance(cls, **kwargs):
        """ creates a test instance used for consistency tests """
        return cls(**cls.get_random_arguments(**kwargs))
  
            
    def ensemble_average(self, method, avg_num=None, multiprocessing=False, 
                         ret_all=False):
        """ calculate an ensemble average of the result of the `method` of
        multiple different receptor libraries """
        
        if avg_num is None:
            avg_num = self.parameters['ensemble_average_num']
        
        if multiprocessing and avg_num > 1:
            # run the calculations in multiple processes  
            arguments = (self.__class__, self.init_arguments, method)
            pool = mp.Pool()
            result = pool.map(_ensemble_average_job, [arguments] * avg_num)
            
            # Apparently, multiprocessing sometimes opens too many files if
            # processes are launched to quickly and the garbage collector cannot
            # keep up. We thus explicitly terminate the pool here.
            pool.terminate()
            
        else:
            # run the calculations in this process
            result = [getattr(self.__class__(**self.init_arguments), method)()
                      for _ in range(avg_num)]
    
        # collect the results and calculate the statistics
        result = np.array(result)
        if ret_all:
            return result
        else:
            return result.mean(axis=0), result.std(axis=0)

    

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
    return getattr(obj, args[2])()

    
    