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


    def _create_test_models(self):
        """ helper method for creating test models """
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
            # set the respective state of the numba patcher
            numba_patcher.set_state(setting['numba_enabled'])
            
            # create test object
            model = LibraryBinaryNumeric.create_test_instance(
                        correlated_mixture=setting['mixture_correlated'],
                        fixed_mixture_size=setting['fixed_mixture_size']
                    )

            # create a meaningful error message for all cases
            model.error_msg = ('The different implementations do not agree for '
                               + ', '.join("%s=%s" % v for v in setting.items()))
            yield model

        # reset numba patcher state
        numba_patcher.set_state(numba_patcher_enabled)


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
                
                
    def test_mixture_entropy(self):
        """ test the calculations of the mixture entropy """
        for model in self._create_test_models():
            error_msg = model.error_msg
            
            # test the mixture entropy calculations
            self.assertAlmostEqual(model.mixture_entropy(),
                                   model.mixture_entropy_brute_force(),
                                   msg='Mixture entropy: ' + error_msg)
            
            self.assertAlmostEqual(model.mixture_entropy(),
                                   model.mixture_entropy_monte_carlo(),
                                   places=1,
                                   msg='Mixture entropy: ' + error_msg)


    def test_mixture_statistics(self):
        """ test mixture statistics calculations """
        for model in self._create_test_models():
            error_msg = model.error_msg
            
            # check the mixture statistics
            ci_1, cij_1 = model.mixture_statistics_brute_force()
            if not model.has_correlations:
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
                
                
    def test_activity_single(self):
        """ test receptor activity calculations """
        for model in self._create_test_models():
            error_msg = model.error_msg
            
            # test activity patterns
            try:
                prob_a_1 = model.activity_single_brute_force()
            except NotImplementedError:
                pass
            else:
                prob_a_2 = model.activity_single_monte_carlo()
                self.assertAllClose(prob_a_1, prob_a_2, rtol=5e-2, atol=5e-2,
                                    msg='Receptor activities: ' + error_msg)
                
                
    def test_activity_correlations(self):
        """ test receptor correlation calculations """
        for model in self._create_test_models():
            error_msg = model.error_msg
            
            # test activity correlation patterns
            try:
                r_nm_1 = model.activity_correlations_brute_force()
            except NotImplementedError:
                pass
            else:
                r_nm_2 = model.activity_correlations_monte_carlo()
                self.assertAllClose(r_nm_1, r_nm_2, rtol=5e-2, atol=5e-2,
                                    msg='Receptor correlations: ' + error_msg)
                
                
    def test_mututal_information(self):
        """ test mutual information calculation """
        for model in self._create_test_models():
            error_msg = model.error_msg
            
            # test calculation of mutual information
            try:
                MI_1 = model.mutual_information_brute_force()
            except NotImplementedError:
                pass
            else:
                MI_2 = model.mutual_information_monte_carlo()
                self.assertAllClose(MI_1, MI_2, rtol=5e-2, atol=5e-2,
                                    msg='Mutual information: ' + error_msg)
    
    
    def test_numba_consistency(self):
        """ test the consistency of the numba functions """
        # this tests the numba consistency for uncorrelated mixtures
        self.assertTrue(numba_patcher.test_consistency(repeat=3, verbosity=1),
                        msg='Numba methods are not consistent')
    
    
    def test_numba_consistency_special(self):
        """ test the consistency of the numba functions """

        # collect all settings that we want to test
        settings = collections.OrderedDict()
        settings['mixture_correlated'] = (True, False)
        settings['fixed_mixture_size'] = (None, 2)
        settings['bias_correction'] = (True, False)
        
        # create all combinations of all settings
        setting_comb = [dict(zip(settings.keys(), items))
                        for items in itertools.product(*settings.values())]
        
        # define the numba methods that need to be tested
        numba_methods = ('LibraryBinaryNumeric.mutual_information_brute_force',
                         'LibraryBinaryNumeric.mutual_information_monte_carlo')
        
        # try all these settings 
        for setting in setting_comb:
            # create a meaningful error message for all cases
            error_msg = ('The Numba implementation is not consistent for ' +
                         ', '.join("%s=%s" % v for v in setting.items()))
            # test the number class
            for name in numba_methods:
                if name.endswith('brute_force') and setting['bias_correction']:
                    continue
                consistent = numba_patcher.test_function_consistency(
                                    name, repeat=2, instance_parameters=setting)
                if not consistent:
                    self.fail(msg=name + '\n' + error_msg)
    
    
    def test_optimization_consistency(self):
        """ test the various optimization methods for consistency """
        
        # list all the tests that should be done
        tests = [
            {'method': 'descent', 'multiprocessing': False},
            {'method': 'descent', 'multiprocessing': True},
            {'method': 'descent_multiple', 'multiprocessing': False},
            {'method': 'descent_multiple', 'multiprocessing': True},
            {'method': 'anneal'},
        ]
        
        # initialize a model
        model = LibraryBinaryNumeric.create_test_instance()
        
        MI_ref = None
        for test_parameters in tests:
            MI, _ = model.optimize_library('mutual_information', direction='max',
                                           steps=1e4, **test_parameters)
            
            if MI_ref is None:
                MI_ref = MI
            else:
                msg = ("Optimization inconsistent (%g != %g) for %s"
                       % (MI_ref, MI, str(test_parameters)))
                self.assertAllClose(MI, MI_ref, rtol=5e-2, atol=5e-2, msg=msg)
        
        

if __name__ == '__main__':
    unittest.main()
    

