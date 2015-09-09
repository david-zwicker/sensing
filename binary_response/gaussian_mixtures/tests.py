'''
Created on May 1, 2015

@author: David Zwicker <dzwicker@seas.harvard.edu>
'''

from __future__ import division

import unittest

import numpy as np

from .lib_gau_base import LibraryGaussianBase
from .numba_speedup import numba_patcher
from ..tests import TestBase



class TestLibraryContinuous(TestBase):
    """ unit tests for the continuous library """

    _multiprocess_can_split_ = True #< let nose know that tests can run parallel
    

    def test_base(self):
        """ consistency tests on the base class """
        # construct random model
        model = LibraryGaussianBase.create_test_instance()
        
        # probability of having d_s components in a mixture for h_i = h
        c_means = model.concentration_means
        for i, c_mean in enumerate(c_means):
            mean_calc = model.get_concentration_distribution(i).mean()
            self.assertAlmostEqual(c_mean, mean_calc)

                
    def test_numba_consistency(self):
        """ test the consistency of the numba functions """
        self.assertTrue(numba_patcher.test_consistency(repeat=3, verbosity=1),
                        msg='Numba methods are not consistent')
                
        
    def _check_histogram(self, observations, distribution, bins=32):
        """ checks whether the observations were likely drawn from the given
        distribution """
        count1, bins = np.histogram(observations, bins=bins, normed=True)
        xs = 0.5*(bins[1:] + bins[:-1])
        count2 = distribution.pdf(xs)
        self.assertAllClose(count1, count2, atol=1e-2, rtol=1e-1,
                            msg='distributions do not agree')
                
               
               
if __name__ == '__main__':
    unittest.main()

