'''
Created on Jan 16, 2015

@author: David Zwicker <dzwicker@seas.harvard.edu>

Module that monkey patches classes in other modules with equivalent, but faster
methods.
'''

from __future__ import division

import functools
import numba
import numpy as np

from .utils import estimate_computation_speed

# these methods are used in getattr calls
import model_numeric  # @UnusedImport

NUMBA_NOPYTHON = True #< globally decide whether we use the nopython mode


@numba.jit(nopython=NUMBA_NOPYTHON)
def ReceptorLibraryNumeric_mutual_information_brute_force_numba(Ns, Nr, sens,
                                                                prob_s, 
                                                                ak, prob_a):
    """ calculate the mutual information by constructing all possible
    mixtures """
    # iterate over all mixtures m
    for m in xrange(2**Ns):
        pm = 1     #< probability of finding this mixture
        ak[:] = 0  #< activity pattern of this mixture
        # iterate through substrates in the mixture
        for i in xrange(Ns):
            r = m % 2
            m //= 2
            if r == 1:
                # substrate i is present
                pm *= prob_s[i]
                for k in xrange(Nr):
                    if sens[k, i] == 1:
                        ak[k] = 1
            else:
                # substrate i is not present
                pm *= 1 - prob_s[i]
                
        # calculate the activity pattern id
        a_id, b = 0, 1
        for k in xrange(Nr):
            if ak[k] == 1:
                a_id += b
            b *= 2
        
        prob_a[a_id] += pm
    
    # calculate the mutual information from the observed probabilities
    MI = 0
    for pa in prob_a:
        if pa > 0:
            MI -= pa*np.log2(pa)
    
    return MI
    


def ReceptorLibraryNumeric_mutual_information_brute_force(self):
    """ updates the energy of the `idx_r`-th receptor """
    return ReceptorLibraryNumeric_mutual_information_brute_force_numba(
        self.Ns, self.Nr, self.sens,
        self.substrate_probability, #< prob_s
        np.empty(self.Nr, np.uint), #< ak
        np.zeros(2**self.Nr) #< prob_a
    )


#===============================================================================
# FUNCTIONS/CLASSES INJECTING THE NUMBA ACCELERATIONS
#===============================================================================


def check_return_value(obj, (func1, func2)):
    """ checks the numba method versus the original one """
    return np.allclose(func1(obj), func2(obj))



class NumbaPatcher(object):
    """ class for managing numba monkey patching in this package. This class
    only provides class methods since it is used as a singleton. """   
    
    # register methods that have a numba equivalent
    numba_methods = {
        'model_numeric.ReceptorLibraryNumeric.mutual_information_brute_force': {
            'numba': ReceptorLibraryNumeric_mutual_information_brute_force,
            'test_function': check_return_value,
            'test_arguments': {},
        },
    }
    
    saved_original_functions = False
    enabled = False #< whether numba speed-up is enabled or not

    
    @classmethod
    def _save_original_function(cls):
        """ save the original function such that they can be restored later """
        for name, data in cls.numba_methods.iteritems():
            module, class_name, method_name = name.split('.')
            class_obj = getattr(globals()[module], class_name)
            data['original'] = getattr(class_obj, method_name)
        cls.saved_original_functions = True


    @classmethod
    def enable(cls):
        """ enables the numba methods """
        if not cls.saved_original_functions:
            cls._save_original_function()
        
        for name, data in cls.numba_methods.iteritems():
            module, class_name, method_name = name.split('.')
            class_obj = getattr(globals()[module], class_name)
            setattr(class_obj, method_name, data['numba'])
        cls.enabled = True
            
            
    @classmethod
    def disable(cls):
        """ disable the numba methods """
        for name, data in cls.numba_methods.iteritems():
            module, class_name, method_name = name.split('.')
            class_obj = getattr(globals()[module], class_name)
            setattr(class_obj, method_name, data['original'])
        cls.enabled = False
        
        
    @classmethod
    def toggle(cls, verbose=True):
        """ enables or disables the numba speed up, depending on the current
        state """
        if cls.enabled:
            cls.disable()
            if verbose:
                print('Numba speed-ups have been disabled.')
        else:
            cls.enable()
            if verbose:
                print('Numba speed-ups have been enabled.')
            
    
    @classmethod
    def _prepare_functions(cls, data):
        """ prepares the arguments for the two functions that we want to test """
        # prepare the arguments
        test_args = data['test_arguments'].copy()
        for key, value in test_args.iteritems():
            if callable(value):
                test_args[key] = value()
                
        # inject the arguments
        func1 = functools.partial(data['original'], **test_args)
        func2 = functools.partial(data['numba'], **test_args)
        return func1, func2

            
    @classmethod
    def test_consistency(cls, repeat=10, verbose=False):
        """ tests the consistency of the numba methods with their original
        counter parts """        
        problems = 0
        for name, data in cls.numba_methods.iteritems():
            # extract the class and the functions
            module, class_name, _ = name.split('.')
            class_obj = getattr(globals()[module], class_name)

            # extract the test function
            test_func = data['test_function']
            
            # check the functions multiple times
            for _ in xrange(repeat):
                test_obj = class_obj.create_test_instance()
                func1, func2 = cls._prepare_functions(data)
                if not test_func(test_obj, (func1, func2)):
                    print('The numba implementation of `%s` is invalid.' % name)
                    print('Native implementation yields %s' % func1(test_obj))
                    print('Numba implementation yields %s' % func2(test_obj))
                    print('Input: %r' % test_obj)
                    problems += 1
                    break
                
            else:
                # there were no problems
                if verbose:
                    print('`%s` has a valid numba implementation.' % name) 

        if not problems:
            print('All numba implementations are consistent.')
            
            
    @classmethod
    def test_speedup(cls, test_duration=1):
        """ tests the speed up of the supplied methods """
        for name, data in cls.numba_methods.iteritems():
            # extract the class and the functions
            module, class_name, func_name = name.split('.')
            class_obj = getattr(globals()[module], class_name)
            test_obj = class_obj.create_test_instance()
            func1, func2 = cls._prepare_functions(data)
                            
            # check the runtime of the original implementation
            speed1 = estimate_computation_speed(func1, test_obj,
                                                test_duration=test_duration)
            # check the runtime of the improved implementation
            speed2 = estimate_computation_speed(func2, test_obj,
                                                test_duration=test_duration)
            
            print('%s.%s: %g times faster' 
                  % (class_name, func_name, speed2/speed1))
            
            