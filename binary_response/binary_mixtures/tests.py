'''
Created on May 1, 2015

@author: David Zwicker <dzwicker@seas.harvard.edu>
'''

from __future__ import division

import unittest

import numpy as np
import scipy.misc

from .library_base import LibraryBinaryBase
      
      
      
class TestLibraryBinary(unittest.TestCase):
    """ unit tests for the continuous library """

    
    def assertAllClose(self, a, b, rtol=1e-05, atol=1e-08, msg=None):
        """ compares all the entries of the arrays a and b """
        self.assertTrue(np.allclose(a, b, rtol, atol), msg)


    def test_base(self):
        """ consistency tests on the base class """
        # construct random model
        model = LibraryBinaryBase.create_test_instance()
        
        # probability of having d_s components in a mixture for h_i = h
        hval = np.random.random() - 0.5
        model.commonness = [hval] * model.Ns
        d_s = np.arange(0, model.Ns + 1)
        p_m = (scipy.misc.comb(model.Ns, d_s)
               * np.exp(hval*d_s)/(1 + np.exp(hval))**model.Ns)
        
        self.assertAllClose(p_m, model.mixture_size_distribution())
    
        # test random commonness and the associated distribution
        hs = np.random.random(size=model.Ns)
        model.commonness = hs
        self.assertAllClose(hs, model.commonness)
        model.substrate_probability = model.substrate_probability
        self.assertAllClose(hs, model.commonness)
        dist = model.mixture_size_distribution()
        self.assertAlmostEqual(dist.sum(), 1)
        ks = np.arange(0, model.Ns + 1)
        dist_mean = (ks*dist).sum()
        dist_var = (ks*ks*dist).sum() - dist_mean**2 
        self.assertAllClose((dist_mean, np.sqrt(dist_var)),
                            model.mixture_size_statistics())
        
        # test setting the commonness
        commoness_schemes = [('const', {}),
                             ('single', {'p1': np.random.random()}),
                             ('single', {'p_ratio': 0.1 + np.random.random()}),
                             ('geometric', {'alpha': np.random.uniform(0.98, 1)}),
                             ('linear', {}),
                             ('random_uniform', {}),]
        
        for scheme, params in commoness_schemes:
            mean_mixture_sizes = (np.random.randint(1, model.Ns//2 + 1),
                                  np.random.randint(1, model.Ns//3 + 1) + model.Ns//2)
            for mean_mixture_size in mean_mixture_sizes:
                model.set_commonness(scheme, mean_mixture_size, **params)
                self.assertAllClose(model.mixture_size_statistics()[0],
                                    mean_mixture_size)

    

if __name__ == '__main__':
    unittest.main()
