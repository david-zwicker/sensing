'''
Created on May 1, 2015

@author: David Zwicker <dzwicker@seas.harvard.edu>
'''

from __future__ import division

import collections
import itertools
import unittest

import numpy as np
import scipy.misc

from .library_base import LibraryBinaryBase
from .library_numeric import LibraryBinaryNumeric
from .numba_speedup import numba_patcher
      
      

class TestLibraryBinary(unittest.TestCase):
    """ unit tests for the continuous library """
    
    _multiprocess_can_split_ = True #< let nose know that tests can run parallel
    
    
    def assertAllClose(self, a, b, rtol=1e-05, atol=1e-08, msg='The two '
                       'arrays do not agree within the given tolerance:'):
        """ compares all the entries of the arrays a and b """
        if not np.allclose(a, b, rtol, atol):
            self.fail(msg + '\nlhs = %s\nrhs = %s' % (a, b))


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
            size1 = np.random.randint(1, model.Ns//2 + 1)
            size2 = np.random.randint(1, model.Ns//3 + 1) + model.Ns//2
            for mean_mixture_size in (size1, size2):
                model.set_commonness(scheme, mean_mixture_size, **params)
                self.assertAllClose(model.mixture_size_statistics()['mean'],
                                    mean_mixture_size)
                
                
    def test_numerics(self):
        """ test numerical calculations """
        # save numba patcher state
        numba_patcher_enabled = numba_patcher.enabled
        
        # collect all settings that we want to test
        settings = collections.OrderedDict()
        settings['numba_enabled'] = (True, False)
        settings['mixture_correlated'] = (True, False)
        settings['fixed_mixture_size'] = (None, 2)
        
        # create all combinations of all settings
        setting_comb = [dict(zip(settings.keys(), items))
                        for items in itertools.product(*settings.values())]
        
        # try all these settings 
        for setting in setting_comb:
            # create a meaningful error message for all cases
            error_msg = ('The different implementations do not agree for ' +
                         ', '.join("%s=%s" % v for v in setting.items()))

            numba_patcher.set_state(setting['numba_enabled'])
            
            # create test object
            model = LibraryBinaryNumeric.create_test_instance(
                        correlated_mixture=setting['mixture_correlated'],
                        fixed_mixture_size=setting['fixed_mixture_size']
                    )

            # test mixture statistics
            ci_1, cij_1 = model.mixture_statistics_brute_force()
            if not setting['mixture_correlated']:
                ci_2, cij_2 = model.mixture_statistics()
                self.assertAllClose(ci_1, ci_2, rtol=5e-2, atol=5e-2,
                                    msg='Mixture statistics: ' + error_msg)
                self.assertAllClose(cij_1, cij_2, rtol=5e-2, atol=5e-2,
                                    msg='Mixture statistics: ' + error_msg)
            ci_2, cij_2 = model.mixture_statistics_monte_carlo()
            self.assertAllClose(ci_1, ci_2, rtol=5e-2, atol=5e-2,
                                msg='Mixture statistics: ' + error_msg)
            self.assertAllClose(cij_1, cij_2, rtol=5e-2, atol=5e-2,
                                msg='Mixture statistics: ' + error_msg)
                
            # test activity patterns
            try:
                prob_a_1 = model.activity_single_brute_force()
            except NotImplementedError:
                pass
            else:
                prob_a_2 = model.activity_single_monte_carlo()
                self.assertAllClose(prob_a_1, prob_a_2, rtol=5e-2, atol=5e-2,
                                    msg='Receptor activities: ' + error_msg)
                
            # test calculation of mutual information
            try:
                MI_1 = model.mutual_information_brute_force()
            except NotImplementedError:
                pass
            else:
                MI_2 = model.mutual_information_monte_carlo()
                self.assertAllClose(MI_1, MI_2, rtol=5e-2, atol=5e-2,
                                    msg='Mutual information: ' + error_msg)
                
        # reset numba patcher state
        numba_patcher.set_state(numba_patcher_enabled)
    
    
    def test_numba_speedup(self):
        """ test the consistency of the numba functions """
        # this tests the numba consistency for uncorrelated mixtures
        self.assertTrue(numba_patcher.test_consistency(1, verbosity=0),
                        msg='Numba methods are not consistent')
    

if __name__ == '__main__':
    unittest.main()
