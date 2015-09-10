'''
Created on May 1, 2015

@author: David Zwicker <dzwicker@seas.harvard.edu>
'''

from __future__ import division

import unittest

from .lib_gau_base import LibraryGaussianBase
from .lib_gau_numeric import LibraryGaussianNumeric
from ..tests import TestBase



class TestLibraryGaussian(TestBase):
    """ unit tests for the Gaussian mixtures """

    _multiprocess_can_split_ = True #< let nose know that tests can run parallel
    

    def _create_test_models(self):
        """ helper method for creating test models """
        # create test object
        model = LibraryGaussianNumeric.create_test_instance()
        model.error_msg = 'Gaussian mixture'
        yield model
        
        
    def test_base(self):
        """ consistency tests on the base class """
        # construct random model
        model = LibraryGaussianBase.create_test_instance()
        
        # probability of having d_s components in a mixture for h_i = h
        c_means = model.concentration_means
        for i, c_mean in enumerate(c_means):
            mean_calc = model.get_concentration_distribution(i).mean()
            self.assertAlmostEqual(c_mean, mean_calc)

                
    def test_estimates(self):
        """ tests the estimates """
        methods = ['concentration_statistics',
                   'excitation_statistics',
                   'receptor_activity']
        
        for model in self._create_test_models():
            error_msg = model.error_msg
            
            # check for known exception where the method are not implemented 
            for method_name in methods:
                method = getattr(model, method_name)
                res_mc = method('monte_carlo')
                res_est = method('estimate')
                
                msg = '%s, Method `%s`' % (error_msg, method_name)
                if method_name.endswith('statistics'):
                    self.assertDictAllClose(res_mc, res_est, rtol=0.1, atol=0.5,
                                            msg=msg)
                else:
                    self.assertAllClose(res_mc, res_est, rtol=0.1, atol=0.5,
                                        msg=msg)
                                        
                                               
               
if __name__ == '__main__':
    unittest.main()

