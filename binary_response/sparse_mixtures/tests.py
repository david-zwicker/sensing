'''
Created on May 1, 2015

@author: David Zwicker <dzwicker@seas.harvard.edu>
'''

from __future__ import division

import collections
import itertools
import unittest

import numpy as np
from scipy import misc
from six.moves import zip_longest

from .lib_spr_base import LibrarySparseBase
from .lib_spr_numeric import LibrarySparseNumeric
from .lib_spr_theory import (LibrarySparseBinary, LibrarySparseLogNormal,
                             LibrarySparseLogUniform)
from .numba_speedup import numba_patcher
from utils.misc import arrays_close

      
      
class TestLibrarySparse(unittest.TestCase):
    """ unit tests for the continuous library """

    _multiprocess_can_split_ = True #< let nose know that tests can run parallel
    
    
    def assertAllClose(self, arr1, arr2, rtol=1e-05, atol=1e-08, msg=None):
        """ compares all the entries of the arrays a and b """
        try:
            # try to convert to numpy arrays
            arr1 = np.asanyarray(arr1)
            arr2 = np.asanyarray(arr2)
            
        except ValueError:
            # try iterating explicitly
            try:
                for v1, v2 in zip_longest(arr1, arr2):
                    self.assertAllClose(v1, v2, rtol, atol, msg)
            except TypeError:
                if msg is None:
                    msg = ""
                else:
                    msg += "; "
                raise TypeError(msg + "Don't know how to compare %s and %s"
                                % (arr1, arr2))
                
        else:
            if msg is None:
                msg = 'Values are not equal'
            msg += '\n%s !=\n%s)' % (arr1, arr2)
            is_close = arrays_close(arr1, arr2, rtol, atol, equal_nan=True)
            self.assertTrue(is_close, msg)

        
    def assertDictAllClose(self, a, b, rtol=1e-05, atol=1e-08, msg=None):
        """ compares all the entries of the dictionaries a and b """
        if msg is None:
            msg = ''
        else:
            msg += '\n'
        
        for k, v in a.iteritems():
            # create a message if non was given
            submsg = msg + ('Dictionaries differ for key `%s` (%s != %s)'
                            % (k, v, b[k]))
                
            # try comparing as numpy arrays and fall back if that doesn't work
            try:
                self.assertAllClose(v, b[k], rtol, atol, submsg)
            except TypeError:
                self.assertEqual(v, b[k], submsg)

        
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
            size1 = np.random.randint(1, model.Ns//2 + 1)
            size2 = np.random.randint(1, model.Ns//3 + 1) + model.Ns//2
            for mean_mixture_size in (size1, size2):
                model.set_commonness(scheme, mean_mixture_size, **params)
                self.assertAllClose(model.mixture_size_statistics()['mean'],
                                    mean_mixture_size)
                

    def test_theory_distributions(self):
        """ test the distributions of the theoretical cases """
        theories = (LibrarySparseBinary.create_test_instance(),
                    LibrarySparseLogNormal.create_test_instance(),
                    LibrarySparseLogUniform.create_test_instance())
        
        for theory in theories:
            stats = theory.sensitivity_stats()
            mean, var = stats['mean'], stats['var']

            if not isinstance(theory, LibrarySparseBinary):            
                dist = theory.sensitivity_distribution
                self.assertAlmostEqual(dist.mean(), mean)
                self.assertAlmostEqual(dist.var(), var)
            
            if not isinstance(theory, LibrarySparseLogUniform):
                theory2 = theory.__class__(theory.Ns, theory.Nr,
                                           mean_sensitivity=mean,
                                           standard_deviation=np.sqrt(var))
                theory2.commonness = theory.commonness
                self.assertDictAllClose(stats, theory2.sensitivity_stats())

        
    def test_theory_limiting(self):
        """ test liming cases of the theory """
        # prepare a random log-normal library
        th1 = LibrarySparseLogNormal.create_test_instance(spread=0.001)
        lib_opt = th1.get_optimal_library(fixed_parameter='spread')
        th1.mean_sensitivity = lib_opt['mean_sensitivity']
        args = th1.init_arguments
        del args['spread']
        del args['correlation']
        th2 = LibrarySparseBinary(**args)
        th2.density = 1
        
        # test various methods on the two libraries
        for method in ['receptor_activity', 'receptor_crosstalk',
                       'mutual_information']:
            res1 = getattr(th1, method)()
            res2 = getattr(th2, method)()
            self.assertAlmostEqual(res1, res2, places=4,
                                   msg='Failed at calculating `%s`' % method)
            
            
    def test_theory_log_normal(self):
        """ test specific functions of the log-normal distribution """
        kept_fixed = ('S0', 'spread')
        for a, b in itertools.permutations(kept_fixed, 2):
            th1 = LibrarySparseLogNormal.create_test_instance()
            lib_opt1 = th1.get_optimal_library(fixed_parameter=a)
            th2 = LibrarySparseLogNormal.from_other(
                th1,
                mean_sensitivity=lib_opt1['mean_sensitivity'],
                spread=lib_opt1['spread']
            )
            lib_opt2 = th2.get_optimal_library(fixed_parameter=b)
            
            msg = 'kept_fixed = (%s, %s)' % (a, b)
            self.assertDictAllClose(lib_opt1, lib_opt2, msg=msg)
    
            
    def test_concentration_statistics(self):
        """ test the statistics of the concentrations """
        model = LibrarySparseNumeric.create_test_instance(num_substrates=128)
        cs = [c.sum() for c in model._sample_mixtures()]
        c_stats = model.concentration_statistics()
        self.assertAllClose(np.mean(cs), c_stats['mean'].sum(), rtol=0.1)
        self.assertAllClose(np.var(cs), c_stats['var'].sum(), rtol=0.1)
            
            
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

                r_nm_calc = np.outer(q_n, q_n) + q_nm
                self.assertAllClose(r_nm, r_nm_calc, rtol=0, atol=0.5,
                                    msg='Receptor correlations: ' + error_msg)
                
    
    def test_estimates(self):
        """ tests the estimates """
        methods = ['excitation_statistics', 'receptor_activity']
        
        for model in self._create_test_models():
            error_msg = model.error_msg
            
            # check for known exception where the method are not implemented 
            for method_name in methods:
                method = getattr(model, method_name)
                res_mc = method('monte_carlo')
                res_est = method('estimate')
                
                msg = '%s, Method `%s`' % (error_msg, method_name)
                if method_name == 'excitation_statistics':
                    self.assertDictAllClose(res_mc, res_est, rtol=0.1, atol=0.5,
                                            msg=msg)
                else:
                    self.assertAllClose(res_mc, res_est, rtol=0.1, atol=0.5,
                                        msg=msg)
                                        
                                
    def test_numba_consistency(self):
        """ test the consistency of the numba functions """
        self.assertTrue(numba_patcher.test_consistency(repeat=3, verbosity=1),
                        msg='Numba methods are not consistent')
        
    

if __name__ == '__main__':
    unittest.main()

