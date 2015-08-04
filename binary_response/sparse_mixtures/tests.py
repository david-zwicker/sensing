'''
Created on May 1, 2015

@author: David Zwicker <dzwicker@seas.harvard.edu>
'''

from __future__ import division

import collections
import itertools
import unittest

import numpy as np
from scipy import misc, stats

from utils.math_distributions import lognorm_mean
from .library_base import LibrarySparseBase  # @UnresolvedImport
from .library_numeric import LibrarySparseNumeric
from .numba_speedup import numba_patcher  # @UnresolvedImport

      
      
class TestLibrarySparse(unittest.TestCase):
    """ unit tests for the continuous library """

    _multiprocess_can_split_ = True #< let nose know that tests can run parallel
    
    
    def assertAllClose(self, a, b, rtol=1e-05, atol=1e-08, msg=None):
        """ compares all the entries of the arrays a and b """
        self.assertTrue(np.allclose(a, b, rtol, atol), msg)


    def test_distributions(self):
        """ test the definiton of parameters of probability distributions """
        S0, sigma = np.random.random(2) + 0.1
        mu = S0 * np.exp(-0.5*sigma**2)
        var = S0**2 * (np.exp(sigma**2) - 1)
        
        # test our distribution and the scipy distribution
        dists = (lognorm_mean(S0, sigma),  stats.lognorm(scale=mu, s=sigma))
        for dist in dists:
            self.assertAlmostEqual(dist.mean(), S0)
            self.assertAlmostEqual(dist.var(), var)
        
        # test the numpy distribution
        rvs = np.random.lognormal(np.log(mu), sigma, size=1000000)
        self.assertAlmostEqual(rvs.mean(), S0, places=3)
        self.assertAlmostEqual(rvs.var(), var, places=3)
        
        
    def _create_test_models(self):
        """ helper method for creating test models """
        # save numba patcher state
        numba_patcher_enabled = numba_patcher.enabled
        
        # collect all settings that we want to test
        settings = collections.OrderedDict()
        settings['numba_enabled'] = (True, False)
        
        # create all combinations of all settings
        setting_comb = [dict(zip(settings.keys(), items))
                        for items in itertools.product(*settings.values())]
        
        # try all these settings 
        for setting in setting_comb:
            # set the respective state of the numba patcher
            numba_patcher.set_state(setting['numba_enabled'])
            
            # create test object
            model = LibrarySparseNumeric.create_test_instance()

            # create a meaningful error message for all cases
            model.error_msg = ('The different implementations do not agree for '
                               + ', '.join("%s=%s" % v for v in setting.items()))
            yield model

        # reset numba patcher state
        numba_patcher.set_state(numba_patcher_enabled)
        

    def test_base(self):
        """ consistency tests on the base class """
        # construct random model
        model = LibrarySparseBase.create_test_instance()
        
        # probability of having m_s components in a mixture for h_i = h
        hval = np.random.random() - 0.5
        model.commonness = [hval] * model.Ns
        m_s = np.arange(0, model.Ns + 1)
        p_m = (misc.comb(model.Ns, m_s)
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
        
        # probability of having m_s components in a mixture for h_i = h
        c_means = model.concentration_means
        for i, c_mean in enumerate(c_means):
            mean_calc = model.get_concentration_distribution(i).mean()
            pi = model.substrate_probabilities[i]
            self.assertAlmostEqual(c_mean, mean_calc * pi)
        
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
                

    def test_correlations_and_crosstalk(self):
        """ tests the correlations and crosstalk """
        for model in self._create_test_models():
            error_msg = model.error_msg
            
            # check for known exception where the method are not implemented 
            for method in ('auto', 'estimate'):
                r_n, r_nm = model.receptor_activity(method, ret_correlations=True) 
                q_n, q_nm = model.receptor_crosstalk(method, ret_receptor_activity=True)
                
                self.assertAllClose(r_n, q_n, rtol=5e-2, atol=5e-2,
                                    msg='Receptor activities: ' + error_msg)
                r_nm_calc = np.clip(np.outer(q_n, q_n) + q_nm, 0, 1)
                self.assertAllClose(r_nm, r_nm_calc, rtol=0, atol=0.5,
                                    msg='Receptor correlations: ' + error_msg)
                
                                
    def test_numba_consistency(self):
        """ test the consistency of the numba functions """
        self.assertTrue(numba_patcher.test_consistency(repeat=3, verbosity=1),
                        msg='Numba methods are not consistent')
                
    

if __name__ == '__main__':
    unittest.main()
