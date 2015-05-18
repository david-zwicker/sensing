'''
Created on May 1, 2015

@author: David Zwicker <dzwicker@seas.harvard.edu>
'''

from __future__ import division

import unittest

import numpy as np
import scipy.stats

from . import LibraryContinuousLogNormal
from .library_base import LibraryContinuousBase
from .numba_speedup import numba_patcher



class TestLibraryContinuous(unittest.TestCase):
    """ unit tests for the continuous library """

    
    def assertAllClose(self, a, b, rtol=1e-05, atol=1e-08, msg=None):
        """ compares all the entries of the arrays a and b """
        self.assertTrue(np.allclose(a, b, rtol, atol), msg)


    def test_base(self):
        """ consistency tests on the base class """
        # construct random model
        model = LibraryContinuousBase.create_test_instance()
        
        # probability of having d_s components in a mixture for h_i = h
        c_means = model.concentration_means
        for i, c_mean in enumerate(c_means):
            mean_calc = model.get_concentration_distribution(i).mean()
            self.assertAlmostEqual(c_mean, mean_calc)


    def test_theory_lognormal_sigma_limit(self):
        """ test some results of the log normal class """
        # check both homogeneous and inhomogeneous mixtures
        for homogeneous, approx in [(True, 'normal'), (False, 'normal'),
                                    (True, 'gamma')]:
            # create random object
            obj1 = LibraryContinuousLogNormal.create_test_instance(
                                               homogeneous_mixture=homogeneous)
            obj2 = obj1.copy()
            
            # compare the results for small spread values (sigma=0 is special case)
            obj1.sigma = 0
            obj2.sigma = 1e-13
            
            # test the activity calculation
            self.assertAlmostEqual(obj1.activity_single_estimate(approx),
                                   obj2.activity_single_estimate(approx),
                                   places=5)
    
            # test the optimal sensitivity calculation
            obj1.typical_sensitivity = \
                    obj1.get_optimal_typical_sensitivity(approximation=approx)
            obj2.typical_sensitivity = \
                    obj2.get_optimal_typical_sensitivity(approximation=approx)
            self.assertAlmostEqual(obj1.typical_sensitivity,
                                   obj2.typical_sensitivity,
                                   places=5)
             
            self.assertAlmostEqual(obj1.activity_single_estimate(approx), 0.5)
            self.assertAlmostEqual(obj2.activity_single_estimate(approx), 0.5)
                
                
    def test_numba_consistency(self):
        """ test the consistency of the numba functions """
        numba_patcher.test_consistency(1, verbosity=0)
                
        
    def _check_histogram(self, observations, distribution):
        """ checks whether the observations were likely drawn from the given
        distribution """
        count1, bins = np.histogram(observations, bins=32, normed=True)
        xs = 0.5*(bins[1:] + bins[:-1])
        count2 = distribution.pdf(xs)
        self.assertAllClose(count1, count2, atol=1e-1, rtol=1e-2,
                            msg='distributions do not agree')
                
                
    def test_log_normal_parameters(self):
        """ test the parameters of the numpy and scipy version of the lognormal
        distribution """
        # choose random parameters
        mean = np.random.random() + .5
        sigma = np.random.random() / 2
        
        # draw from and define distribution        
        ys = np.random.lognormal(mean=np.log(mean), sigma=sigma, size=int(1e7))
        dist = scipy.stats.lognorm(scale=mean, s=sigma)
        self._check_histogram(ys, dist)
        
        # compare to standard definition of the pdf
        xs = np.linspace(0, 10)[1:]
        norm = xs * sigma * np.sqrt(2*np.pi)
        ys1 = dist.pdf(xs)
        ys2 = np.exp(-np.log(xs/mean)**2/(2*sigma**2)) / norm
        self.assertAllClose(ys1, ys2, msg='pdf of scipy lognorm is different '
                                          'from expected one')
                
                
    def test_gamma_parameters(self):
        """ test the parameters of the numpy and scipy version of the gamma
        distribution """
        # choose random parameters
        count = np.random.randint(5, 10)
        mean = np.random.random() + .5

        # get the scipy distribution
        dist = scipy.stats.gamma(a=count, scale=mean)
        
        # draw multiple exponential random numbers
        ci = np.random.exponential(size=(count, int(1e5))) * mean
        ctot = ci.sum(axis=0)
        self._check_histogram(ctot, dist)
        
        # draw from and define distribution    
        ys = np.random.gamma(count, mean, size=int(1e5))    
        self._check_histogram(ys, dist)
        
        # compare to standard definition of the pdf
        xs = np.linspace(0, 20)[1:]
        ys1 = dist.pdf(xs)
        ys2 = (xs**(count - 1) 
               * np.exp(-xs / mean)
               / mean**count
               / scipy.special.gamma(count))  # @UndefinedVariable
        self.assertAllClose(ys1, ys2, msg='pdf of scipy gamma distribution is '
                                          'different from expected one')
               
               
               
if __name__ == '__main__':
    unittest.main()

