'''
Created on Mar 27, 2015

@author: zwicker
'''

from __future__ import division

import itertools
import time
import multiprocessing as mp

import numpy as np



class ReceptorLibraryNumeric(object):
    """ represents a single receptor library """
    
    monte_carlo_steps = 10000  #< default number of monte carlo steps
    

    def __init__(self, num_substrates, num_receptors, hs, frac=1):
        """ initialize the receptor library by setting the number of receptors,
        the number of substrates it can respond to, the weights `hs` of the 
        substrates, and the fraction `frac` of substrates a single receptor
        responds to """
        assert len(hs) == num_substrates
        assert num_receptors < 63 #< prevent integer overflow
        
        self.Ns = num_substrates
        self.Nr = num_receptors
        self.hs = hs
        self.frac = frac
        
        np.random.seed()
        self.choose_sensitivites()


    @property
    def parameters(self):
        """ return the parameters of the model that can be used to reconstruct
        it by calling the __init__ method with these arguments """
        return (self.Ns, self.Nr, self.hs, self.frac)


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
            
            
    def activity_single_monte_carlo(self, num=None):
        """ calculates the average activity of each receptor """ 
        if num is None:
            num = self.monte_carlo_steps        
    
        # calculate the probability of seeing each substrate independently
        prob_h = np.exp(self.hs)/(1 + np.exp(self.hs))
        
        count_a = np.zeros(self.Nr)
        for _ in xrange(num):
            # choose a mixture vector according to substrate probabilities
            m = (np.random.random(self.Ns) < prob_h)
            
            # get the associated output ...
            a = np.dot(self.sens, m).astype(np.bool)
            
            count_a[a] += 1
            
        return count_a/num

            
    def mutual_information_brute_force(self, ret_prob_activity=False):
        """ calculate the mutual information by constructing all possible
        mixtures """
        base = 2 ** np.arange(self.Nr-1, -1, -1)

        # calculate the probability of seeing each substrate independently
        prob_h = np.exp(self.hs)/(1 + np.exp(self.hs))

        # prob_a contains the probability of finding activity a as an output.
        prob_a = np.zeros(2**self.Nr)
        for m in itertools.product((0, 1), repeat=self.Ns):
            # get the associated output ...
            a = np.dot(self.sens, m).astype(np.bool)
            # ... and represent it as a single integer
            a = np.dot(base, a)

            # probability of finding this substrate
            ma = np.array(m, np.bool)
            pm = np.prod(prob_h[ma]) * np.prod(1 - prob_h[~ma])
            prob_a[a] += pm
        
        # calculate the mutual information
        MI = -sum(pa*np.log(pa) for pa in prob_a if pa != 0)
        
        if ret_prob_activity:
            return MI, prob_a.mean()
        else:
            return MI
            
            
    def mutual_information_monte_carlo(self, num=None, ret_prob_activity=False):
        """ calculate the mutual information by sampling `num` mixtures. If 
        `num` is not given, the class variable `monte_carlo_steps` is used. """
        if num is None:
            num = self.monte_carlo_steps        
        
        base = 2 ** np.arange(self.Nr-1, -1, -1)

        # calculate the probability of seeing each substrate independently
        prob_h = np.exp(self.hs)/(1 + np.exp(self.hs))
        
        count_a = np.zeros(2**self.Nr)
        for _ in xrange(num):
            # choose a mixture vector according to substrate probabilities
            m = (np.random.random(self.Ns) < prob_h)
            
            # get the associated output ...
            a = np.dot(self.sens, m).astype(np.bool)
            # ... and represent it as a single integer
            a = np.dot(base, a)
            # increment counter for this output
            count_a[a] += 1
            
        # count_a contains the number of times output pattern a was observed.
        # We can thus construct P_a(a) from count_a. 

        prob_a = count_a / num
        MI = -sum(pa*np.log(pa) for pa in prob_a if pa != 0)

        if ret_prob_activity:
            return MI, prob_a.mean()
        else:
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


    def average_mutual_information_mp(self, method='monte_carlo', avg_num=32):
        """ calculate a mean mutual information by averaging over `avg_num`
        different sensitivity matrices. Uses multiprocessing """
        # run the calculations in multiple processes  
        pool = mp.Pool()
        arguments = (method, self.Nr, self.Ns, self.hs, self.frac)
        MIs = pool.map(_ReceptorLibrary_mp_calc_MI, [arguments] * avg_num)

        # collect the results and calculate the statistics
        MIs = np.array(MIs)
        return MIs.mean(), MIs.std()



def _ReceptorLibrary_mp_calc_MI(args):
    """ helper function for multiprocessing """
    method = args[0]
    obj = ReceptorLibraryNumeric(*args[1:])
    if method == 'monte_carlo':
        return obj.mutual_information_monte_carlo()
    elif method == 'brute_force':
        return obj.mutual_information_brute_force()
    else:
        raise ValueError('Unknown method `%s`' % method)



def _ReceptorLibrary_mp_calc(args):
    """ helper function for multiprocessing """
    obj = ReceptorLibraryNumeric(*args[0])
    return getattr(obj, args[1])()



def ensemble_average_mp(model, method, avg_num=32, ret_all=False):
    """ calculate an ensemble average of the result of the `method` of multiple
    different receptor libraries using the parameters of `model` """
    # run the calculations in multiple processes  
    pool = mp.Pool()
    arguments = (model.parameters, method)
    result = pool.map(_ReceptorLibrary_mp_calc, [arguments] * avg_num)

    # collect the results and calculate the statistics
    result = np.array(result)
    if ret_all:
        return result
    else:
        return result.mean(axis=0), result.std(axis=0)

   
   
def performance_test(Ns=15, Nr=3, frac=0.5):
    """ test the performance of the brute force and the monte carlo method """
    num = 2**Ns
    hs = np.random.random(Ns)
    model = ReceptorLibraryNumeric(Nr, Ns, hs, frac=frac)
    
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
    

    
