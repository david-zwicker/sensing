'''
Created on Mar 27, 2015

@author: zwicker
'''

from __future__ import division

import itertools
import time

import numpy as np


class ReceptorLibrary(object):
    """ represents a single receptor library """


    def __init__(self, num_receptors, num_substrates, hs, frac=1):
        """ initialize the receptor library by setting the number of receptors,
        the number of substrates it can respond to, the weights `hs` of the 
        substrates, and the fraction `frac` of substrates a single receptor
        responds to """
        assert len(hs) == num_substrates
        assert num_receptors < 63 #< prevent integer overflow
        
        self.Nr = num_receptors
        self.Ns = num_substrates
        self.hs = hs
        self.frac = frac
        
        self.choose_sensitivites()


    def choose_sensitivites(self):
        """ creates a sensitivity matrix """
        shape = (self.Nr, self.Ns)
        # choose receptor substrate interaction randomly
        self.sens = (np.random.random(shape) < self.frac).astype(np.int)
            
            
    def mixture_size_distribution(self):
        """ calculates the probabilities of finding a mixture with a given
        number of components. Returns an array of length Ns + 1 of probabilities
        for finding mixtures with the number of components given by the index
        into the array """
        # calculate the probability of seeing each substrate independently
        prob_h = np.exp(self.hs)/(1 + np.exp(self.hs))
        res = np.zeros(self.Ns + 1)
        res[0] = 1
        for k, p in enumerate(prob_h, 1):
            r = res[:k].copy()
            res[:k] *= 1 - p  #< substrate not in the mixture 
            res[1:k+1] += r*p #< substrate in the mixture
            
        return res
            
            
    def mixture_size_statistics(self):
        """ calculates the mean and the standard deviation of the number of
        components in mixtures """
        exp_h = np.exp(self.hs)
        denom = 1 + exp_h
        l_mean = np.sum(exp_h/denom)
        l_var = np.sum(exp_h/denom**2)
        
        return l_mean, np.sqrt(l_var)
            
            
    def mutual_information_brute_force(self):
        """ calculate the mutual information by constructing all possible
        mixtures """
        base = 2 ** np.arange(self.Nr-1, -1, -1)

        # calculate the probability of seeing each substrate independently
        prob_h = np.exp(self.hs)/(1 + np.exp(self.hs))

        count_ma = np.zeros(2**self.Nr, np.uint)
        prob_a = np.zeros(2**self.Nr)
        for m in itertools.product((0, 1), repeat=self.Ns):
            # get the associated output ...
            a = np.dot(self.sens, m) > 0.5
            # ... and represent it as a single integer
            a = np.dot(base, a)

            # probability of finding this substrate
            ma = np.array(m, np.bool)
            pm = np.prod(prob_h[ma]) * np.prod(1 - prob_h[~ma])
            count_ma[a] += 1
            prob_a[a] += pm
            
        # count_ma contains the counts of how many mixtures m map to the same
        # activity a. prob_a contains the probability of finding activity a
        # as an output.
        
        # calculate the mutual information
        MI = -sum(p_a*np.log(p_a)
                  for c_ma, p_a in itertools.izip(count_ma, prob_a)
                  if c_ma > 0)
        
        return MI
            
            
    def mutual_information_monte_carlo(self, num=1000):
        """ calculate the mutual information by sampling `num` mixtures """
        base = 2 ** np.arange(self.Nr-1, -1, -1)

        # calculate the probability of seeing each substrate independently
        prob_h = np.exp(self.hs)/(1 + np.exp(self.hs))
        
        count_a = np.zeros(2**self.Nr)
        for _ in xrange(num):
            # choose a mixture vector according to substrate probabilities
            m = (np.random.random(self.Ns) < prob_h)
            
            # get the associated output
            a = np.dot(self.sens, m) > 0.5
            # represent it as a single integer
            a = np.dot(base, a)
            # count how often each output occurs
            count_a[a] += 1
            
        # count_a contains the number of times output pattern a was observed.
        # We can thus construct P_a(a) from count_a. 

        prob_a = count_a[count_a > 0] / num
        MI = -np.sum(prob_a*np.log(prob_a))

        return MI
    
    
    def average_mutual_information(self, method='monte_carlo', avg_num=32):
        """ calculate a mean mutual information by averaging over `avg_num`
        different sensitivity matrices """
        if method == 'monte_carlo':
            calc_MI = self.mutual_information_monte_carlo
        elif method == 'brute_force':
            calc_MI = self.mutual_information_brute_force
        else:
            raise ValueError('Unknown method `%s`' % method)
        
        MIs = np.empty(avg_num)
        for k in xrange(avg_num):
            self.choose_sensitivites()
            MIs[k] = calc_MI()
        
        return MIs.mean(), MIs.std()

            
            
def performance_test(Ns=15, Nr=3, frac=0.5):
    """ test the performance of the brute force and the monte carlo method """
    num = 2**Ns
    hs = np.random.random(Ns)
    model = ReceptorLibrary(Nr, Ns, hs, frac=frac)
    
    start = time.time()
    model.simulate_brute_force()
    time_brute_force = time.time() - start
    print('Brute force: %g sec' % time_brute_force)
    
    start = time.time()
    model.simulate_monte_carlo(num)
    time_monte_carlo = time.time() - start
    print('Monte carlo: %g sec' % time_monte_carlo)
    
    
            
if __name__ == '__main__':
    performance_test()
    

    