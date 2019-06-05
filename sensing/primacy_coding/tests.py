'''
Created on Dec 31, 2015

@author: David Zwicker <dzwicker@seas.harvard.edu>
'''

from __future__ import division

import collections
import itertools
import unittest

import numpy as np
from scipy import special

from .pc_numeric import PrimacyCodingNumeric, nlargest_indices
from .pc_theory import PrimacyCodingTheory
from .numba_speedup_numeric import numba_patcher, nlargest_indices_numba
from utils.testing import TestBase 

      

class TestLibraryPrimacyCoding(TestBase):
    """ unit tests for the continuous library """

    _multiprocess_can_split_ = True #< let nose know that tests can run parallel
    
    
    def test_nlargest_indices(self):
        """ test a helper function """
        arr = np.arange(10)
        self.assertEqual(set(nlargest_indices(arr, 2)), set([8, 9]))
        
        arr = np.random.randn(128)
        self.assertEqual(set(nlargest_indices(arr, 32)),
                         set(nlargest_indices_numba(arr, 32)))
    
        
    def _create_test_models(self, **kwargs):
        """ helper method for creating test models """
        # save numba patcher state
        numba_patcher_enabled = numba_patcher.enabled
        
        # collect all settings that we want to test
        settings = collections.OrderedDict()
        settings['numba_enabled'] = (True, False)
        c_dists = PrimacyCodingNumeric.concentration_distributions
        settings['c_distribution'] = c_dists
        
        # create all combinations of all settings
        setting_comb = [dict(zip(settings.keys(), items))
                        for items in itertools.product(*settings.values())]
        
        # try all these settings 
        for setting in setting_comb:
            # set the respective state of the numba patcher
            numba_patcher.set_state(setting['numba_enabled'])
            
            # create test object
            model = PrimacyCodingNumeric.create_test_instance(**kwargs)

            # create a meaningful error message for all cases
            model.settings = ', '.join("%s=%s" % v for v in setting.items())
            model.error_msg = ('The different implementations do not agree for '
                               + model.settings)
            yield model

        # reset numba patcher state
        numba_patcher.set_state(numba_patcher_enabled)
        

    def test_base(self):
        """ consistency tests on the base class """
        # construct random model
        model = PrimacyCodingNumeric.create_test_instance()
        
        # probability of having m_s components in a mixture for h_i = h
        hval = np.random.random() - 0.5
        model.commonness = [hval] * model.Ns
        m_s = np.arange(0, model.Ns + 1)
        p_m = (special.comb(model.Ns, m_s)
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
            size1 = np.random.randint(1, model.Ns//2 + 1)
            size2 = np.random.randint(1, model.Ns//3 + 1) + model.Ns//2
            for mean_mixture_size in (size1, size2):
                model.choose_commonness(scheme, mean_mixture_size, **params)
                self.assertAllClose(model.mixture_size_statistics()['mean'],
                                    mean_mixture_size)


    def test_theory(self):
        """ test some theory """
        # create test instance with many receptors
        theory = PrimacyCodingTheory.create_test_instance(num_receptors=300,
                                                          coding_receptors=10)
        
        for excitation_dist in ('gaussian', 'log-normal'):
            theory.parameters['excitation_distribution'] = excitation_dist
            
            # compare the excitation thresholds
            et1 = theory.excitation_threshold(method='approx', corr_term=0)[0]
            et2 = theory.excitation_threshold(method='integrate', corr_term=0)[0]
            
            msg = ('The calculated excitation thresholds differ. (%g != %g)'
                   % (et1, et2))
            self.assertAllClose(et1, et2, rtol=0.05, msg=msg)


    def test_theory_numba_tb(self):
        """ test numba code in the theory part for consistency """
        from . import pc_theory
        numba_func = pc_theory._activity_distance_tb_lognorm_integrand_numba
        
        # create test instance
        theory = PrimacyCodingTheory.create_test_instance(num_receptors=30,
                                                          coding_receptors=5)
        # test different excitation models
        for excitation_dist in ('gaussian', 'log-normal'):
            theory.parameters['excitation_distribution'] = excitation_dist
            # test different concentration ratios
            for c1 in [0.1, 1, 10]:
                pc_theory._activity_distance_tb_lognorm_integrand_numba = None
                h1 = theory.activity_distance_target_background(c1)
                pc_theory._activity_distance_tb_lognorm_integrand_numba = numba_func
                h2 = theory.activity_distance_target_background(c1)
                self.assertAllClose(h1, h2, msg='Numba code not consistent')


#     def test_theory_numba_m(self):
#         """ test numba code in the theory part for consistency """
#         from . import pc_theory
#         numba_func = pc_theory._activity_distance_m_lognorm_integrand_numba
#         
#         # create test instance
#         theory = PrimacyCodingTheory.create_test_instance(num_receptors=30,
#                                                           coding_receptors=5)
#         # test different excitation models
#         for excitation_dist in ('gaussian', 'log-normal'):
#             theory.parameters['excitation_distribution'] = excitation_dist
#             # test different concentration ratios
#             for sB in [0, 5, 10]:
#                 pc_theory._activity_distance_m_lognorm_integrand_numba = None
#                 h1 = theory.activity_distance_mixtures(10, sB)
#                 pc_theory._activity_distance_m_lognorm_integrand_numba = numba_func
#                 h2 = theory.activity_distance_mixtures(10, sB)
#                 self.assertAllClose(h1, h2, msg='Numba code not consistent')


    def test_setting_coding_receptors(self):
        """ tests whether setting the coding_receptors is consistent """
        # construct specific model
        params = {'coding_receptors': 6}
        model = PrimacyCodingNumeric(6, 6, parameters=params)
        self.assertEqual(model.coding_receptors, 6)
        self.assertEqual(model.parameters['coding_receptors'], 6)
        
        # construct random model
        model = PrimacyCodingNumeric.create_test_instance(coding_receptors=5)
        self.assertEqual(model.coding_receptors, 5)
        self.assertEqual(model.parameters['coding_receptors'], 5)
        
        # change coding_receptors
        model.coding_receptors = 4
        self.assertEqual(model.coding_receptors, 4)
        self.assertEqual(model.parameters['coding_receptors'], 4)


    def test_numba_consistency(self):
        """ test the consistency of the numba functions """
        self.assertTrue(numba_patcher.test_consistency(repeat=3, verbosity=1),
                        msg='Numba methods are not consistent')
        
    

if __name__ == '__main__':
    unittest.main()

