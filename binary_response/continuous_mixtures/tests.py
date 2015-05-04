'''
Created on May 1, 2015

@author: David Zwicker <dzwicker@seas.harvard.edu>
'''

from __future__ import division

import unittest

from . import LibraryContinuousLogNormal
from .library_base import LibraryContinuousBase



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
        # create random object
        obj1 = LibraryContinuousLogNormal.create_test_instance()
        obj2 = LibraryContinuousLogNormal(**obj1.init_arguments)
        
        # set the spread value
        obj1.sigma = 0
        obj2.sigma = 1e-10
        
        self.assertAlmostEqual(obj1.activity_single(), obj2.activity_single())

    

if __name__ == '__main__':
    unittest.main()

