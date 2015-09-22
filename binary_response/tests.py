'''
Created on May 1, 2015

@author: David Zwicker <dzwicker@seas.harvard.edu>
'''

from __future__ import division

import unittest

import numpy as np
from six.moves import zip_longest

from .library_base import LibraryBase
from utils.misc import arrays_close

      
      
class TestBase(unittest.TestCase):
    """ extends the basic TestCase class with some convenience functions """ 
      
    def assertAllClose(self, arr1, arr2, rtol=1e-05, atol=1e-08, msg=None):
        """ compares all the entries of the arrays a and b """
        try:
            # try to convert to numpy arrays
            arr1 = np.asanyarray(arr1)
            arr2 = np.asanyarray(arr2)
            
        except ValueError:
            # try iterating explicitly
            try:
                for v1, v2 in zip_longest(arr1, arr2):
                    self.assertAllClose(v1, v2, rtol, atol, msg)
            except TypeError:
                if msg is None:
                    msg = ""
                else:
                    msg += "; "
                raise TypeError(msg + "Don't know how to compare %s and %s"
                                % (arr1, arr2))
                
        else:
            if msg is None:
                msg = 'Values are not equal'
            msg += '\n%s !=\n%s)' % (arr1, arr2)
            is_close = arrays_close(arr1, arr2, rtol, atol, equal_nan=True)
            self.assertTrue(is_close, msg)

        
    def assertDictAllClose(self, a, b, rtol=1e-05, atol=1e-08, msg=None):
        """ compares all the entries of the dictionaries a and b """
        if msg is None:
            msg = ''
        else:
            msg += '\n'
        
        for k, v in a.items():
            # create a message if non was given
            submsg = msg + ('Dictionaries differ for key `%s` (%s != %s)'
                            % (k, v, b[k]))
                
            # try comparing as numpy arrays and fall back if that doesn't work
            try:
                self.assertAllClose(v, b[k], rtol, atol, submsg)
            except TypeError:
                self.assertEqual(v, b[k], submsg)
      
      
      
class TestLibraryBase(TestBase):
    """ unit tests for the continuous library """

    _multiprocess_can_split_ = True #< let nose know that tests can run parallel
    

    def test_base_class(self):
        """ test the base class """
        obj = LibraryBase.create_test_instance()
        
        # calculate mutual information
        for method in ('expansion', 'hybrid', 'polynom'):
            if method == 'polynom':
                # we can consider heterogeneous receptor response
                q_n = 0.1 + 0.8*np.random.rand(obj.Nr)
                q_nm = 0.1*np.random.rand(obj.Nr, obj.Nr)
        
            else:
                # we have to have homogeneous receptor response
                q_n = np.zeros(obj.Nr) + np.random.rand()
                q_nm = np.zeros((obj.Nr, obj.Nr)) + 0.1*np.random.rand()

            np.fill_diagonal(q_nm, 0)
            q_nm_var = q_nm[~np.eye(obj.Nr, dtype=np.bool)].var()
            
            MI1 = obj._estimate_MI_from_q_values(q_n, q_nm, method=method)
            MI2 = obj._estimate_MI_from_q_stats(
                q_n.mean(), q_nm.mean(), q_n.var(), q_nm_var,
                method=method
            )
            msg = 'Mutual informations do not agree for method=`%s`' % method
            self.assertAllClose(MI1, MI2, rtol=0.1, msg=msg)
                    
    

if __name__ == '__main__':
    unittest.main()
