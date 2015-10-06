'''
Created on Aug 21, 2015

@author: David Zwicker <dzwicker@seas.harvard.edu>
'''

from __future__ import division

import unittest

import numpy as np
from scipy import stats 

from .math_distributions import (lognorm_mean_var_to_mu_sigma, lognorm_mean,
                                 lognorm_mean_var, loguniform_mean)

      
      
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
            lognorm_mean(np.random.random() + 0.1, np.random.random() + 0.1),
            lognorm_mean_var(np.random.random() + 0.1, np.random.random() + 0.2),
            loguniform_mean(np.random.random() + 0.1, np.random.random() + 1.1),
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
        dists = (lognorm_mean(S0, sigma), stats.lognorm(scale=mu, s=sigma))
        for dist in dists:
            self.assertAlmostEqual(dist.mean(), S0)
            self.assertAlmostEqual(dist.var(), var)
        
        # test the numpy distribution
        rvs = np.random.lognormal(np.log(mu), sigma, size=1000000)
        self.assertAlmostEqual(rvs.mean(), S0, places=2)
        self.assertAlmostEqual(rvs.var(), var, places=1)

        # test the numpy distribution
        mean, var = np.random.random() + 0.1, np.random.random() + 0.1
        dist = lognorm_mean(mean, var)
        self.assertAlmostEqual(dist.mean(), mean)
        dist = lognorm_mean_var(mean, var)
        self.assertAlmostEqual(dist.mean(), mean)
        self.assertAlmostEqual(dist.var(), var)
        
        mu, sigma = lognorm_mean_var_to_mu_sigma(mean, var, 'numpy')
        rvs = np.random.lognormal(mu, sigma, size=1000000)
        self.assertAlmostEqual(rvs.mean(), mean, places=2)
        self.assertAlmostEqual(rvs.var(), var, places=1)


    def test_log_uniform(self):
        """ test the log uniform distribution """
        S0 = np.random.random() + 0.1
        sigma = np.random.random() + 1.1
        
        # test our distribution and the scipy distribution
        dist = loguniform_mean(S0, sigma)
        self.assertAlmostEqual(dist.mean(), S0)
                    
    

if __name__ == '__main__':
    unittest.main()
