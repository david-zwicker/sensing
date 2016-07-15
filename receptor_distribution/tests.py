'''
Created on Jul 15, 2016

@author: zwicker
'''

from __future__ import division

import unittest
import numpy as np

from .receptor_distribution import ExcitationProfiles, ReceptorDistribution
from utils.tests import TestBase



class Test(TestBase):
    """ manage tests of the receptor distribution """

    _multiprocess_can_split_ = True #< let nose know that tests can run parallel


    def setUp(self):
        self.Nr = np.random.randint(5, 10)
        self.num_x = np.random.randint(5, 10)

        # choose random excitation profiles
        self.excitations = ExcitationProfiles(self.Nr, self.num_x)
        self.excitations.choose_exponential_excitations(
                            np.random.random() + 0.5, np.random.random() + 0.5)
         
        # choose random receptors
        self.receptors = ReceptorDistribution(self.Nr, self.num_x)
    

    def test_distribution_conversion(self):
        dist_reduced = np.random.random(size=(self.Nr - 1, self.num_x))
        
        dist_real = self.receptors._distribution_reduced2real(dist_reduced)
        self.assertAllClose(dist_real.sum(axis=0), np.ones(self.num_x))
        
        dist_reduced_calc = self.receptors._distribution_real2reduced(dist_real)
        self.assertAllClose(dist_reduced, dist_reduced_calc)



if __name__ == "__main__":
    unittest.main()