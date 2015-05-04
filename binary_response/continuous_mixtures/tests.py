'''
Created on May 1, 2015

@author: David Zwicker <dzwicker@seas.harvard.edu>
'''

from __future__ import division

import unittest

from . import LibraryContinuousLogNormal
from .library_base import LibraryContinuousBase
from .numba_speedup import numba_patcher



class TestLibraryContinuous(unittest.TestCase):
    """ unit tests for the continuous library """

    def test_base(self):
        """ consistency tests on the base class """
        # construct random model
        model = LibraryContinuousBase.create_test_instance()
        
        # probability of having d_s components in a mixture for h_i = h
        c_means = model.get_concentration_means()
        for i, c_mean in enumerate(c_means):
            mean_calc = model.get_concentration_distribution(i).mean()
            self.assertAlmostEqual(c_mean, mean_calc)


    def test_theory_lognormal(self):
        """ test some results of the log normal class """
        # check both homogeneous and inhomogeneous mixtures
        for homogeneous in (True, False):
            # create random object
            obj1 = LibraryContinuousLogNormal.create_test_instance(
                                               homogeneous_mixture=homogeneous)
            obj2 = obj1.copy()
            
            # compare the results for small spread values (sigma=0 is special case)
            obj1.sigma = 0
            obj2.sigma = 1e-10
    
            # test the activity calculation
            self.assertAlmostEqual(obj1.activity_single(),
                                   obj2.activity_single())
    
            # test the optimal sensitivity calculation
            obj1.mean_sensitivity = obj1.get_optimal_mean_sensitivity()
            obj2.mean_sensitivity = obj2.get_optimal_mean_sensitivity()
            self.assertAlmostEqual(obj1.mean_sensitivity, obj2.mean_sensitivity)
            
            self.assertAlmostEqual(obj1.activity_single(), 0.5)
            self.assertAlmostEqual(obj2.activity_single(), 0.5)
                
                
    def test_numba_speedup(self):
        """ test the consistency of the numba functions """
        numba_patcher.test_consistency(1, verbosity=0)
                
     

if __name__ == '__main__':
    unittest.main()

