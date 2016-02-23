'''
Created on May 1, 2015

@author: David Zwicker <dzwicker@seas.harvard.edu>
'''

from __future__ import division

from contextlib import contextmanager
import functools

import numpy as np
from six.moves import reduce

from .misc import estimate_computation_speed, copy_func 



def check_return_value(obj, funcs):
    """ checks the numba method versus the original one """
    val1, val2 = funcs[0](obj), funcs[1](obj)
    result = np.allclose(val1, val2)
    return result, (val1, val2)


def check_return_value_approx(obj, funcs, rtol=5e-2, atol=5e-2):
    """ checks the numba method versus the original one """
    val1, val2 = funcs[0](obj), funcs[1](obj)
    result = np.allclose(val1, val2, rtol=rtol, atol=atol)
    return result, (val1, val2)


def check_return_dict_approx(obj, funcs, rtol=0.1, atol=0.1):
    """ checks the numba method versus the original one """
    val1, val2 = funcs[0](obj), funcs[1](obj)
    result = all(np.allclose(val1[key], val2[key], rtol=rtol, atol=atol)
                 for key in val1), (val1, val2)
    return result, (val1, val2)


def check_return_value_exact(obj, funcs):
    """ checks the numba method versus the original one """
    return check_return_value_approx(obj, funcs, rtol=1e-10, atol=1e-14)



class NumbaPatcher(object):
    """ class for managing numba monkey patching. """   
    
    def __init__(self, module=None):
        """ initialize to patch functions and classes in module `module` """
        self.module = module
        self.numba_methods = {}
        self.saved_python_functions = False
        self.enabled = False #< whether numba speed-up is enabled or not
    

    @property
    def disabled(self):
        return not self.enabled


    def register_method(self, name, numba_function,
                        test_function=check_return_value, test_arguments=None):
        """ register a new numba method """
        if test_arguments is None:
            test_arguments = {}
        numba_function = copy_func(numba_function)
        self.numba_methods[name] = {'numba_function': numba_function,
                                    'test_function': test_function,
                                    'test_arguments': test_arguments}


    def _resolve_object(self, class_name):
        """ resolves nested class definitions """
        return reduce(getattr, class_name.split('.'), self.module)

    
    def _save_python_function(self):
        """ save the original function such that they can be restored later """
        for name, data in self.numba_methods.items():
            python_function = self._resolve_object(name)
            data['python_function'] = python_function
            data['numba_function']._python_function = python_function
        self.saved_python_functions = True


    def enable(self):
        """ enables the numba methods """
        old_state = self.enabled
        
        if not self.saved_python_functions:
            self._save_python_function()
        
        for name, data in self.numba_methods.items():
            class_name, method_name = name.rsplit('.', 1)
            class_obj = self._resolve_object(class_name)
            setattr(class_obj, method_name, data['numba_function'])
        self.enabled = True
        
        return old_state
            
            
    def disable(self):
        """ disable the numba methods """
        old_state = self.enabled

        for name, data in self.numba_methods.items():
            class_name, method_name = name.rsplit('.', 1)
            class_obj = self._resolve_object(class_name)
            setattr(class_obj, method_name, data['python_function'])
        self.enabled = False
        
        return old_state
        
        
    def toggle(self, verbose=True):
        """ enables or disables the numba speed up, depending on the current
        state """
        if self.enabled:
            self.disable()
            if verbose:
                print('Numba speed-ups have been disabled.')
        else:
            self.enable()
            if verbose:
                print('Numba speed-ups have been enabled.')
            
    
    def set_state(self, enable=True):
        """ sets the state to the given value """
        if enable:
            self.enable()
        else:
            self.disable()
        
            
    @contextmanager
    def as_disabled(self):
        """ context manager for temporarily disabling the patcher and restoring
        the previous state afterwards """
        if self.enabled:
            self.disable()
            yield
            self.enable()
        else:
            yield

    
    def _prepare_functions(self, data):
        """ prepares the arguments for the two functions that we want to test """
        # prepare the arguments
        test_args = data['test_arguments'].copy()
        for key, value in test_args.items():
            if callable(value):
                test_args[key] = value()
                
        # inject the arguments
        func1 = functools.partial(data['python_function'], **test_args)
        func2 = functools.partial(data['numba_function'], **test_args)
        return func1, func2

            
    def test_function_consistency(self, name, repeat=3, verbosity=1,
                                  instance_parameters=None):
        """ tests the consistency of a single numba methods with their original
        counter part.
        `verbosity` controls how verbose the output is going to be. Valid values
            are 0, 1, 2 with increasing verbosity, respectively.
        """        

        # extract the class and the functions
        class_name, _ = name.split('.')
        class_obj = getattr(self.module, class_name)
        
        data = self.numba_methods[name]

        # extract the test function
        test_func = data['test_function']
        
        # extract the test instance parameters
        if instance_parameters is None:
            instance_parameters = {}
        
        # check the functions multiple times
        for _ in range(repeat):
            test_obj = class_obj.create_test_instance(**instance_parameters)
            func1, func2 = self._prepare_functions(data)
            consistent, vals = test_func(test_obj, (func1, func2))
            if not consistent:
                print('The numba implementation of `%s` is invalid.' % name)
                print('Native implementation yields %s' % str(vals[0]))
                print('Numba implementation yields %s' % str(vals[1]))
                print('Input: %r' % test_obj)
                break
            
        else:
            # there were no problems
            if verbosity >= 2:
                print('`%s` has a valid numba implementation.' % name)
            return True
                
        # we exited the for loop and there thus was a problem
        return False
                     
                                
    def test_consistency(self, repeat=10, verbosity=2):
        """ tests the consistency of the numba methods with their original
        counter parts.
        `verbosity` controls how verbose the output is going to be. Valid values
            are 0, 1, 2 with increasing verbosity, respectively.
        """        
        # test all registered numba functions
        for name in self.numba_methods:
            consistent = self.test_function_consistency(name, repeat, verbosity)
            if not consistent:
                # we found a mistake and return immediately
                if verbosity >= 1:
                    print('Numba method `%s` is not consistent.' % name) 
                return False

        # we did not find a mistake and signal that to the caller
        if verbosity >= 2:
            print('All numba implementations are consistent.')
        
        return True
            
            
    def test_speedup(self, test_duration=1):
        """ tests the speed up of the supplied methods """
        for name, data in self.numba_methods.items():
            # extract the class and the functions
            class_name, func_name = name.split('.')
            class_obj = getattr(self.module, class_name)
            test_obj = class_obj.create_test_instance()
            func1, func2 = self._prepare_functions(data)
                            
            # check the runtime of the original implementation
            speed1 = estimate_computation_speed(func1, test_obj,
                                                test_duration=test_duration)
            # check the runtime of the improved implementation
            speed2 = estimate_computation_speed(func2, test_obj,
                                                test_duration=test_duration)
            
            print('%s.%s: %g times faster' 
                  % (class_name, func_name, speed2/speed1))
            
            