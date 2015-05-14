'''
Created on Jan 21, 2015

@author: David Zwicker <dzwicker@seas.harvard.edu>
'''

from __future__ import division

import contextlib
import functools
import itertools
import sys
import timeit
import types
import warnings
from collections import Counter

import numpy as np
from scipy.stats import itemfreq


def score_interaction_matrices(I1, I2):
    """ returns a score of the similarity of the interaction matrices, taking
    into account all permutations of the receptors """
    assert I1.shape == I2.shape
    
    return min(np.abs(I1[perm, :] - I2).mean()
               for perm in itertools.permutations(range(len(I1))))



def estimate_computation_speed(func, *args, **kwargs):
    """ estimates the computation speed of a function """
    test_duration = kwargs.pop('test_duration', 1)
    
    # prepare the function
    if args or kwargs:
        test_func = functools.partial(func, *args, **kwargs)
    else:
        test_func = func
    
    # call function once to allow caches be filled
    test_func()
     
    # call the function until the total time is achieved
    number, duration = 1, 0
    while duration < 0.1*test_duration:
        number *= 10
        duration = timeit.timeit(test_func, number=number)
    return number/duration



def get_fastest_entropy_function():
    """ returns a function that calculates the entropy of a array of integers
    Here, several alternative definitions are tested and the fastest one is
    returned """ 
    def entropy_numpy(arr):
        """ entropy function based on numpy.unique """
        fs = np.unique(arr, return_counts=True)[1]
        return np.sum(fs*np.log2(fs))
    def entropy_scipy(arr):
        """ entropy function based on scipy.stats.itemfreq """
        fs = itemfreq(arr)[:, 1]
        return np.sum(fs*np.log2(fs))
    def entropy_counter1(arr):
        """ entropy function based on collections.Counter """
        return sum(val*np.log2(val)
                   for val in Counter(arr).itervalues())
    def entropy_counter2(arr):
        """ entropy function based on collections.Counter """
        return sum(val*np.log2(val)
                   for val in Counter(arr).values())

    test_array = np.random.random_integers(0, 10, 100)
    func_fastest, speed_max = None, 0
    for test_func in (entropy_numpy, entropy_scipy, entropy_counter1,
                      entropy_counter2):
        try:
            speed = estimate_computation_speed(test_func, test_array)
        except (TypeError, AttributeError):
            # TypeError: older numpy versions don't support `return_counts`
            # AttributeError: python3 does not have iteritems
            pass
        else:
            if speed > speed_max:
                func_fastest, speed_max = test_func, speed

    return func_fastest

calc_entropy = get_fastest_entropy_function()



class classproperty(object):
    """ decorator that can be used to define read-only properties for classes. 
    Code copied from http://stackoverflow.com/a/5192374/932593
    """
    def __init__(self, f):
        self.f = f
        
    def __get__(self, obj, owner):
        return self.f(owner)
    
    
    
class DummyFile(object):
    """ dummy file that ignores all write calls """
    def write(self, x):
        pass



@contextlib.contextmanager
def silent_stdout():
    """
    context manager that silence the standard output
    Code copied from http://stackoverflow.com/a/2829036/932593
    """
    save_stdout = sys.stdout
    sys.stdout = DummyFile()
    yield
    sys.stdout = save_stdout
    
    

def copy_func(f, name=None):
    """ copies a python function. Taken from
    http://stackoverflow.com/a/6528148/932593
    """ 
    return types.FunctionType(f.func_code, f.func_globals, name or f.func_name,
                              f.func_defaults, f.func_closure)



class DeprecationHelper(object):
    """
    Helper function for re-routing deprecated classes 
    copied from http://stackoverflow.com/a/9008509/932593
    """
    
    def __init__(self, new_target, warning_class=Warning):
        self.new_target = new_target
        self.warning_class = warning_class

    def _warn(self):
        msg = "The class was renamed to `%s`"  % self.new_target.__name__
        warnings.warn(msg, self.warning_class, stacklevel=3)

    def __call__(self, *args, **kwargs):
        self._warn()
        return self.new_target(*args, **kwargs)

    def __getattr__(self, attr):
        self._warn()
        return getattr(self.new_target, attr)
    
    