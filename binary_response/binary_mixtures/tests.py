'''
Created on May 1, 2015

@author: David Zwicker <dzwicker@seas.harvard.edu>
'''

from __future__ import division

import itertools
import unittest

import numpy as np
import scipy.misc

from .library_base import LibraryBinaryBase
from .library_numeric import LibraryBinaryNumeric
from .numba_speedup import numba_patcher
      
      
      
class TestLibraryBinary(unittest.TestCase):
    """ unit tests for the continuous library """
    
    def assertAllClose(self, a, b, rtol=1e-05, atol=1e-08, msg=None):
        """ compares all the entries of the arrays a and b """
        if not np.allclose(a, b, rtol, atol):
            print('lhs = %s' % a)
            print('rhs = %s' % b)
            self.fail(msg)


    def test_base(self):
        """ consistency tests on the base class """
        # construct random model
        model = LibraryBinaryBase.create_test_instance()
        
        # probability of having m_s components in a mixture for h_i = h
        hval = np.random.random() - 0.5
        model.commonness = [hval] * model.Ns
        m_s = np.arange(0, model.Ns + 1)
        p_m = (scipy.misc.comb(model.Ns, m_s)
               * np.exp(hval*m_s)/(1 + np.exp(hval))**model.Ns)
        
        self.assertAllClose(p_m, model.mixture_size_distribution())
    
        # test random commonness and the associated distribution
        hs = np.random.random(size=model.Ns)
        model.commonness = hs
        self.assertAllClose(hs, model.commonness)
        model.substrate_probabilities = model.substrate_probabilities
        self.assertAllClose(hs, model.commonness)
        dist = model.mixture_size_distribution()
        self.assertAlmostEqual(dist.sum(), 1)
        ks = np.arange(0, model.Ns + 1)
        dist_mean = (ks*dist).sum()
        dist_var = (ks*ks*dist).sum() - dist_mean**2
        stats = model.mixture_size_statistics() 
        self.assertAllClose((dist_mean, dist_var),
                            (stats['mean'], stats['var']))
        
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
                self.assertAllClose(model.mixture_size_statistics()['mean'],
                                    mean_mixture_size)
                
                
    def test_correlations(self):
        """ test mixtures with correlations """
        for correlated in (True, False):
            # create test object
            model = LibraryBinaryNumeric.create_test_instance(
                                                  correlated_mixture=correlated)
            
            # extract parameters
            hi = model.commonness
            Jij = model.correlations
            
            # calculate the mean correlations 
            ci_exact = np.zeros(model.Ns)
            cij_exact = np.zeros((model.Ns, model.Ns))
            Z = 0
            for c in itertools.product((0, 1), repeat=model.Ns):
                c = np.array(c)
                prob_c = np.exp(np.dot(hi - np.dot(Jij, c), c))
                Z += prob_c
                ci_exact += c * prob_c
                cij_exact += np.outer(c, c) * prob_c
            
            ci_exact /= Z
            cij_exact /= Z
            cij_corr_exact = cij_exact - np.outer(ci_exact, ci_exact)
            
            # calculate this numerically
            ci_mean_numeric, cij_corr_numeric = model.mixture_statistics()
            self.assertAllClose(ci_exact, ci_mean_numeric)
            self.assertAllClose(cij_corr_exact, cij_corr_numeric)

                
    def test_numerics(self):
        """ test numerical calculations """
        for correlated in (True, False):
            # create test object
            model = LibraryBinaryNumeric.create_test_instance(
                                                  correlated_mixture=correlated)

            # test activity patterns
            prob_a_1 = model.activity_single_brute_force()
            if not correlated:
                prob_a_2 = model.activity_single_monte_carlo()
                self.assertAllClose(prob_a_1, prob_a_2, rtol=1e-2, atol=1e-2)
            prob_a_2 = model.activity_single_metropolis()
            self.assertAllClose(prob_a_1, prob_a_2, rtol=1e-2, atol=1e-2)
                
            # test calculation of mutual information
            prob_a_1 = model.mutual_information_brute_force()
            if not correlated:
                prob_a_2 = model.mutual_information_monte_carlo()
                self.assertAllClose(prob_a_1, prob_a_2, rtol=1e-2, atol=1e-2)
            prob_a_2 = model.mutual_information_metropolis()
            self.assertAllClose(prob_a_1, prob_a_2, rtol=1e-2, atol=1e-2)
                
                
    def test_numba_speedup(self):
        """ test the consistency of the numba functions """
        # this tests the numba consistency for uncorrelated mixtures
        self.assertTrue(numba_patcher.test_consistency(1, verbosity=0))
    

if __name__ == '__main__':
    unittest.main()
