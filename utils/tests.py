'''
Created on Aug 21, 2015

@author: David Zwicker <dzwicker@seas.harvard.edu>
'''

from __future__ import division

import unittest

import numpy as np
from scipy import stats 
from six.moves import zip_longest

from . import math_distributions
from .misc import arrays_close
from .numba_tools import lognorm_cdf, lognorm_pdf

      

      
class TestBase(unittest.TestCase):
    """ extends the basic TestCase class with some convenience functions """ 
      
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
        
        for k, v in a.items():
            # create a message if non was given
            submsg = msg + ('Dictionaries differ for key `%s` (%s != %s)'
                            % (k, v, b[k]))
                
            # try comparing as numpy arrays and fall back if that doesn't work
            try:
                self.assertAllClose(v, b[k], rtol, atol, submsg)
            except TypeError:
                self.assertEqual(v, b[k], submsg)
      
            
      
class TestMathDistributions(unittest.TestCase):
    """ unit tests for the continuous library """

    _multiprocess_can_split_ = True #< let nose know that tests can run parallel
    
    
    def assertAllClose(self, a, b, rtol=1e-05, atol=1e-08, msg=None):
        """ compares all the entries of the arrays a and b """
        self.assertTrue(np.allclose(a, b, rtol, atol), msg)


    def test_dist_rvs(self):
        """ test random variates """
        # create some distributions to test
        distributions = [
            math_distributions.lognorm_mean(np.random.random() + 0.1,
                                            np.random.random() + 0.1),
            math_distributions.lognorm_mean_var(np.random.random() + 0.1,
                                                np.random.random() + 0.2),
            math_distributions.loguniform_mean(np.random.random() + 0.1,
                                               np.random.random() + 1.1),
        ]
        
        # calculate random variates and compare them to the given mean and var.
        for dist in distributions:
            rvs = dist.rvs(100000)
            self.assertAllClose(dist.mean(), rvs.mean(), rtol=0.02,
                                msg='Mean of the distribution is not '
                                    'consistent.')
            self.assertAllClose(dist.var(), rvs.var(), rtol=0.2, atol=0.1,
                                msg='Variance of the distribution is not '
                                    'consistent.')


    def test_log_normal(self):
        """ test the log normal distribution """
        S0, sigma = np.random.random(2) + 0.1
        mu = S0 * np.exp(-0.5*sigma**2)
        var = S0**2 * (np.exp(sigma**2) - 1)
        
        # test our distribution and the scipy distribution
        dists = (math_distributions.lognorm_mean(S0, sigma),
                 stats.lognorm(scale=mu, s=sigma))
        for dist in dists:
            self.assertAlmostEqual(dist.mean(), S0)
            self.assertAlmostEqual(dist.var(), var)
        
        # test the numpy distribution
        rvs = np.random.lognormal(np.log(mu), sigma, size=1000000)
        self.assertAlmostEqual(rvs.mean(), S0, places=2)
        self.assertAlmostEqual(rvs.var(), var, places=1)

        # test the numpy distribution
        mean, var = np.random.random() + 0.1, np.random.random() + 0.1
        dist = math_distributions.lognorm_mean(mean, var)
        self.assertAlmostEqual(dist.mean(), mean)
        dist = math_distributions.lognorm_mean_var(mean, var)
        self.assertAlmostEqual(dist.mean(), mean)
        self.assertAlmostEqual(dist.var(), var)
        
        mu, sigma = math_distributions.lognorm_mean_var_to_mu_sigma(mean, var,
                                                                    'numpy')
        rvs = np.random.lognormal(mu, sigma, size=1000000)
        self.assertAlmostEqual(rvs.mean(), mean, places=2)
        self.assertAlmostEqual(rvs.var(), var, places=1)


    def test_gamma(self):
        """ test the log uniform distribution """
        mean = np.random.random() + 0.1
        var = np.random.random() + 1.1
        
        # test our distribution and the scipy distribution
        dist = math_distributions.gamma_mean_var(mean, var)
        self.assertAlmostEqual(dist.mean(), mean)
        self.assertAlmostEqual(dist.var(), var)
                    
                    
    def test_log_uniform(self):
        """ test the log uniform distribution """
        S0 = np.random.random() + 0.1
        sigma = np.random.random() + 1.1
        
        # test our distribution and the scipy distribution
        dist = math_distributions.loguniform_mean(S0, sigma)
        self.assertAlmostEqual(dist.mean(), S0)
                    
                    
    def test_numba_stats(self):
        """ test the numba implementation of statistics functions """
        for _ in range(10):
            mean = np.random.random() + 0.1
            var = np.random.random() + 0.1
            x = np.random.random() + 0.1
            dist_LN = math_distributions.lognorm_mean_var(mean, var)
            self.assertAlmostEqual(dist_LN.pdf(x), lognorm_pdf(x, mean, var))
            self.assertAlmostEqual(dist_LN.cdf(x), lognorm_cdf(x, mean, var))
    
    

if __name__ == '__main__':
    unittest.main()
