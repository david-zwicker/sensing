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


def xlog2x(x):
    """ calculates x*np.log2(x) """
    if x == 0:
        return 0
    else:
        return x * np.log2(x)

# vectorize the function above
xlog2x = np.vectorize(xlog2x, otypes='d')



def arrays_close(arr1, arr2, rtol=1e-05, atol=1e-08, equal_nan=False):
    """ compares two arrays using a relative and an absolute tolerance """
    arr1 = np.atleast_1d(arr1)
    arr2 = np.atleast_1d(arr2)
    
    if arr1.shape != arr2.shape:
        # arrays with different shape are always unequal
        return False
        
    if equal_nan:
        # skip entries where both arrays are nan
        idx = ~(np.isnan(arr1) & np.isnan(arr2))
        if idx.sum() == 0:
            # occurs when both arrays are full of NaNs
            return True

        arr1 = arr1[idx]
        arr2 = arr2[idx]
    
    # get the scale of the first array
    scale = np.linalg.norm(arr1.flat, np.inf)
    
    # try to compare the arrays
    with np.errstate(invalid='raise'):
        try:
            is_close = np.any(np.abs(arr1 - arr2) <= (atol + rtol * scale))
        except FloatingPointError:
            is_close = False
        
    return is_close



def is_pos_semidef(x):
    """ checks whether the correlation matrix is positive semi-definite """
    return np.all(np.linalg.eigvals(x) >= 0)



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
    
    # define a bunch of functions that act the same but have different speeds 
    
    def entropy_numpy(arr):
        """ calculate the entropy of the distribution given in `arr` """
        fs = np.unique(arr, return_counts=True)[1]
        return np.sum(fs*np.log2(fs))
    
    def entropy_scipy(arr):
        """ calculate the entropy of the distribution given in `arr` """
        fs = itemfreq(arr)[:, 1]
        return np.sum(fs*np.log2(fs))
    
    def entropy_counter1(arr):
        """ calculate the entropy of the distribution given in `arr` """
        return sum(val*np.log2(val)
                   for val in Counter(arr).itervalues())
        
    def entropy_counter2(arr):
        """ calculate the entropy of the distribution given in `arr` """
        return sum(val*np.log2(val)
                   for val in Counter(arr).values())

    # test all functions against a random array to find the fastest one
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



def calc_entropy(arr):
    """ calculate the entropy of the distribution given in `arr` """
    # find the fastest entropy function on the first call of this function
    # and bind it to the same name such that it is used in future times
    global calc_entropy
    calc_entropy = get_fastest_entropy_function()
    return calc_entropy(arr)



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
    
    

class CachedArray(object):
    def __init__(self, value=None):
        self._data = np.empty(0)
        self.value = value
    
    def __call__(self, shape):
        if self._data.shape == shape:
            if self.value is not None:
                self._data.fill(self.value)
        else:
            if self.value is None:
                self._data = np.empty(shape)
            elif self.value == 0:
                self._data = np.zeros(shape)
            else: 
                self._data = np.full(shape, self.value, np.double)
        return self._data
