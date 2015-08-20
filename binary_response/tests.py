'''
Created on May 1, 2015

@author: David Zwicker <dzwicker@seas.harvard.edu>
'''

from __future__ import division

import unittest

import numpy as np

from library_base import LibraryBase  # @UnresolvedImport

      
      
class TestLibraryBase(unittest.TestCase):
    """ unit tests for the continuous library """

    _multiprocess_can_split_ = True #< let nose know that tests can run parallel
    
    
    def assertAllClose(self, a, b, rtol=1e-05, atol=1e-08, msg=None):
        """ compares all the entries of the arrays a and b """
        self.assertTrue(np.allclose(a, b, rtol, atol), msg)


    def test_base_class(self):
        """ test the base class """
        obj = LibraryBase.create_test_instance()
        
        # prepare fictious receptor response
        q_n = np.random.rand(obj.Nr)
        q_nm = 0.1*np.random.rand(obj.Nr, obj.Nr)
        np.fill_diagonal(q_nm, 0)
        
        # calculate mutual information
        MI1 = obj._estimate_mutual_information_from_q_values(q_n, q_nm)
        MI2 = obj._estimate_mutual_information_from_q_stats(
                                q_n.mean(), q_nm.mean(), q_n.var(), q_nm.var())
        self.assertAllClose(MI1, MI2, rtol=0.1)
                    
    

if __name__ == '__main__':
    unittest.main()
