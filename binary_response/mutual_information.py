'''
Created on Mar 27, 2015

@author: zwicker
'''

from __future__ import division

import collections
import random

import numpy as np


class ModelNumeric(object):
    """ a simple energies that implements the binary receptors response energies,
    where each concentration field in the mucus is mapped to a set of binary
    numbers indicating whether the respective receptors are active or not """

    _label = 'Numeric'


    def __init__(self, num_receptors, num_substrates, hs, frac=1):
        """ initialize a energies for calculation of num_odors odorants
        `energies` defines the energies used for the energies
        `num_receptors` determines the number of receptors        
        """
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
        self.sens = (np.random.random(shape) > self.frac).astype(np.int)
            
            
    def mixture_statistics(self):
        """ calculates statistics of mixtures """
        assert len(np.unique(self.hs)) == 1
        exph = np.exp(self.hs[0])
        
        l_mean = (exph*self.Ns)/(1 + exph)
        l_var = (exph*self.Ns)/(1 + exph)**2
        
        return l_mean, np.sqrt(l_var)
            
            
    def simulate(self, num=10000):
        """ run the simulation """
        
        base = 2 ** np.arange(self.Nr, 0, -1)

        # calculate the probability of seeing each substrate independently
        prob_h = np.exp(self.hs)/(1 + np.exp(self.hs))
        
        pm_norm = 0
        res = np.zeros((num, 2))
        for k in xrange(num):
            # choose a mixture vector according to substrate probabilities
            m = (np.random.random(self.Ns) < prob_h)
            
            # get the output 
            a = np.dot(self.sens, m) > 0.5
            # represent it as a single integer
            a = np.dot(base, a)

            # keep track of the output
            pm = np.exp(np.dot(self.hs, m))
            pm_norm += pm
            res[k, 0] = pm
            res[k, 1] = a

        # get the probability distribution of finding an output a  
        prob_a = collections.Counter(res[:, 1])
        
        # calculate the mutual information
        MI = -sum(prob_m * np.log(prob_a[a]/num)
                  for prob_m, a in res)/pm_norm
        
        return MI


    def simulate_old(self, num=1000):
        """ run the simulation """
        
        base = 2 ** np.arange(self.Nr, 0, -1)

        # start with a random mixture
        m = (np.random.random(self.Ns) > 0.5).astype(np.int)
        pm = np.exp(np.dot(self.hs, m))
        
        pm_norm = 0
        res = np.zeros((num, 2))
        k = 0
        while True:
            # create new test
            m_test = m[:]
            i = random.randrange(self.Ns)
            m_test[i] = 1 - m_test[i]
            
            # calculate the probability of this mixture
            pm_test = np.exp(np.dot(self.hs, m_test))

            # calculate acceptance probability            
            alpha = pm_test/pm
            
            if alpha >= 1 or random.random() < alpha:
                m = m_test
                pm = pm_test
                
                pm_norm += pm #< keep track of the normalization
                # get the output and represent it as a single integer
                a = np.dot(base, np.dot(self.sens, m) > 0.5)

                # keep track of the output
                res[k, 0] = pm
                res[k, 1] = a
                
                
                # prepare next step
                k += 1
                
                if k >= num:
                    break
              
        # get the probability distribution of finding an output a  
        prob_a = collections.Counter(res[:, 1])
        
        # calculate the mutual information
        MI = -sum(prob_m * np.log(prob_a[a]/num)
                  for prob_m, a in res)/pm_norm
        
        return MI
            
            
if __name__ == '__main__':
    res = []
    for frac in np.linspace(0, 1, 10):
        model = ModelNumeric(5, 20, np.zeros(20) - 2, frac=frac)
        #print model.mixture_statistics()
        res.append(model.simulate())
        
    import matplotlib.pyplot as plt
    plt.plot(res)
    plt.show()

    