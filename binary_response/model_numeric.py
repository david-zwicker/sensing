'''
Created on Mar 27, 2015

@author: zwicker

General note on multiprocessing
-------------------------------
Some of the functions support multiprocessing to distribute a calculation among
several processors. If this is used, it is best practice to safe-guard the main
program with the following construct

if __name__ == '__main__': 
    <main code of the program>

This is also explained in
https://docs.python.org/2/library/multiprocessing.html#using-a-pool-of-workers
'''

from __future__ import division

import copy
import itertools
import time
import multiprocessing as mp
import random

import numpy as np

from simanneal import Annealer

from .model_base import ReceptorLibraryBase



class ReceptorLibraryNumeric(ReceptorLibraryBase):
    """ represents a single receptor library """
    
    parameters_default = {
        'max_num_receptors': 28,    #< prevents memory overflows
        'random_seed': None,        #< seed for the random number generator
        'sensitivity_matrix': None, #< will be calculated if not given
        'monte_carlo_steps': 10000, #< default number of monte carlo steps
        'monte_carlo_strategy': 'frequency',
        'anneal_Tmax': 1e0,         #< Max (starting) temperature for annealing
        'anneal_Tmin': 1e-3,        #< Min (ending) temperature for annealing
        'verbosity': 1,             #< verbosity level    
    }
    

    def __init__(self, num_substrates, num_receptors, hs=None, frac=1,
                 parameters=None):
        """ initialize the receptor library by setting the number of receptors,
        the number of substrates it can respond to, the weights `hs` of the 
        substrates, and the fraction `frac` of substrates a single receptor
        responds to """
        super(ReceptorLibraryNumeric, self).__init__(num_substrates,
                                                     num_receptors, hs, frac)        

        # set the internal parameters
        self.parameters = self.parameters_default.copy()
        if parameters is not None:
            self.parameters.update(parameters)
        
        # prevent integer overflow in collecting activity patterns
        assert num_receptors < 63 
        assert num_receptors <= self.parameters['max_num_receptors']
        
        np.random.seed(self.parameters['random_seed'])
        self.choose_sensitivites()


    @property
    def init_arguments(self):
        """ return the parameters of the model that can be used to reconstruct
        it by calling the __init__ method with these arguments """
        return {'num_substrates': self.Ns,
                'num_receptors': self.Nr,
                'hs': self._hs,
                'frac': self.frac,
                'parameters': self.parameters}


    def choose_sensitivites(self):
        """ creates a sensitivity matrix """
        shape = (self.Nr, self.Ns)
        # choose receptor substrate interaction randomly
        if self.parameters['sensitivity_matrix'] is None:
            self.sens = (np.random.random(shape) < self.frac).astype(np.int)
        else:
            self.sens = self.parameters['sensitivity_matrix']
            assert self.sens.shape == shape
            
            
    def sort_sensitivities(self, sensitivities=None):
        """ return the sorted `sensitivities` or sorts the internal
        sensitivities in place """
        sens = self.sens if sensitivities is None else sensitivities

        data = [(sum(item), list(item)) for item in sens]
        sens = np.array([item[1] for item in sorted(data)])

        if sensitivities is None:
            self.sens = sens
        else:
            return sens
            
            
    def activity_single_monte_carlo(self, num=None):
        """ calculates the average activity of each receptor """ 
        if num is None:
            num = int(self.parameters['monte_carlo_steps'])        
    
        prob_s = self.substrate_probability
    
        count_a = np.zeros(self.Nr)
        for _ in xrange(num):
            # choose a mixture vector according to substrate probabilities
            m = (np.random.random(self.Ns) < prob_s)
            
            # get the associated output ...
            a = np.dot(self.sens, m).astype(np.bool)
            
            count_a[a] += 1
            
        return count_a/num

            
    def mutual_information_brute_force(self, ret_prob_activity=False):
        """ calculate the mutual information by constructing all possible
        mixtures """
        base = 2 ** np.arange(self.Nr-1, -1, -1)

        prob_s = self.substrate_probability

        # prob_a contains the probability of finding activity a as an output.
        prob_a = np.zeros(2**self.Nr)
        for m in itertools.product((0, 1), repeat=self.Ns):
            # get the associated output ...
            a = np.dot(self.sens, m).astype(np.bool)
            # ... and represent it as a single integer
            a = np.dot(base, a)

            # probability of finding this substrate
            ma = np.array(m, np.bool)
            pm = np.prod(prob_s[ma]) * np.prod(1 - prob_s[~ma])
            prob_a[a] += pm
        
        # calculate the mutual information
        MI = -sum(pa*np.log2(pa) for pa in prob_a if pa != 0)
        
        if ret_prob_activity:
            return MI, prob_a.mean()
        else:
            return MI
            
            
    def mutual_information_monte_carlo(self, num=None, ret_prob_activity=False):
        """ calculate the mutual information by strategy `num` mixtures. If 
        `num` is not given, the parameter `monte_carlo_steps` is used. """
        if num is None:
            num = int(self.parameters['monte_carlo_steps'])
                
        base = 2 ** np.arange(self.Nr-1, -1, -1)
        prob_s = self.substrate_probability

        strategy = self.parameters['monte_carlo_strategy']
        if strategy == 'frequency':
            # sample mixtures according to the probabilities of finding
            # substrates
            count_a = np.zeros(2**self.Nr)
            for _ in xrange(num):
                # choose a mixture vector according to substrate probabilities
                m = (np.random.random(self.Ns) < prob_s)
                
                # get the associated output ...
                a = np.dot(self.sens, m).astype(np.bool)
                # ... and represent it as a single integer
                a = np.dot(base, a)
                # increment counter for this output
                count_a[a] += 1
                
            # count_a contains the number of times output pattern a was observed.
            # We can thus construct P_a(a) from count_a. 
    
            prob_a = count_a / num
            
        elif strategy == 'uniform':
            # sample mixtures with each substrate being equally likely and
            # correct the probabilities 
            prob_a = np.zeros(2**self.Nr)
            for _ in xrange(num):
                # choose a mixture vector according to substrate probabilities
                m = np.random.randint(2, size=self.Ns)
                
                # get the associated output ...
                a = np.dot(self.sens, m).astype(np.bool)
                # ... and represent it as a single integer
                a = np.dot(base, a)

                # probability of finding this substrate
                ma = np.array(m, np.bool)
                pm = np.prod(prob_s[ma]) * np.prod(1 - prob_s[~ma])
                prob_a[a] += pm
                
            # normalize the probabilities    
            prob_a /= prob_a.sum()
            
        else:
            raise ValueError('Unknown strategy strategy `%s`' % strategy)
            
        # calculate the mutual information from the result pattern
        MI = -sum(pa*np.log2(pa) for pa in prob_a if pa != 0)

        if ret_prob_activity:
            return MI, prob_a.mean()
        else:
            return MI
    
    
    def ensemble_average(self, method, avg_num=32, multiprocessing=False, 
                         ret_all=False):
        """ calculate an ensemble average of the result of the `method` of
        multiple different receptor libraries """
        
        if multiprocessing:
            # run the calculations in multiple processes  
            arguments = (self.init_arguments, method)
            pool = mp.Pool()
            result = pool.map(_ReceptorLibrary_mp_calc, [arguments] * avg_num)
            
        else:
            # run the calculations in this process
            result = [getattr(ReceptorLibraryNumeric(**self.init_arguments),
                              method)()
                      for _ in xrange(avg_num)]
    
        # collect the results and calculate the statistics
        result = np.array(result)
        if ret_all:
            return result
        else:
            return result.mean(axis=0), result.std(axis=0)
        
        
    def optimize_library(self, target, method='descent', steps=100, 
                         ret_info=False):
        """ optimizes the current library to maximize the result of the target
        function. By default, the function returns the best value and the
        associated sensitivity matrix as result.        
        
        `steps` determines how many optimization steps we try 
        `ret_info` determines whether extra information is returned from the
            optimization 

        `method` determines the method used for optimization. Supported are
            `descent`: simple gradient descent for `steps` number of steps
            `descent_parallel`: multiprocessing gradient descent. Note that this
                has an overhead and might actually decrease overall performance
                for small problems.
            `anneal`: simulated annealing
        """
        if method == 'descent':
            return self.optimize_library_descent(target, steps,
                                                 multiprocessing=False,
                                                 ret_info=ret_info)
        elif method == 'descent_parallel':
            return self.optimize_library_descent(target, steps,
                                                 multiprocessing=True,
                                                 ret_info=ret_info)
        elif method == 'anneal':
            return self.optimize_library_anneal(target, steps, ret_info)
            
        else:
            raise ValueError('Unknown optimization method `%s`' % method)
            
        
    def optimize_library_descent(self, target, steps=100, multiprocessing=False, 
                                 ret_info=False):
        """ optimizes the current library to maximize the result of the target
        function using gradient descent. By default, the function returns the
        best value and the associated sensitivity matrix as result.        
        
        `steps` determines how many optimization steps we try
        `multiprocessing` is a flag deciding whether multiple processes are used
            to calculate the result. Note that this has an overhead and might
            actually decrease overall performance for small problems
        `ret_info` determines whether extra information is returned from the
            optimization 
        """
        # get the target function to call
        target_function = getattr(self, target)

        # initialize the optimizer
        value = target_function()
        value_best, state_best = value, self.sens.copy()
        if ret_info:
            info = {'values': []}
        
        if multiprocessing:
            # run the calculations in multiple processes
            pool = mp.Pool()
            pool_size = len(pool._pool)
            for _ in xrange(steps // pool_size):
                joblist = []
                init_arguments = self.init_arguments
                for _ in xrange(pool_size):
                    # modify the current state and add it to the job list
                    i = random.randrange(self.sens.size)
                    self.sens.flat[i] = 1 - self.sens.flat[i]
                    init_arguments['parameters']['sensitivity_matrix'] = self.sens
                    joblist.append((copy.deepcopy(init_arguments), target))
                    self.sens.flat[i] = 1 - self.sens.flat[i]
                    
                # run all the jobs
                results = pool.map(_ReceptorLibrary_mp_calc, joblist)
                
                # find the best result                
                res_best = np.argmax(results)
                if results[res_best] > value_best:
                    value_best = results[res_best]
                    state_best = joblist[res_best][0]['parameters']['sensitivity_matrix']
                    # use the best state as a basis for the next iteration
                    self.sens = state_best
                if ret_info:
                    info['values'].append(results[res_best])
                
        else:  
            # run the calculations in this process
            for _ in xrange(steps):
                # modify the current state
                i = random.randrange(self.sens.size)
                self.sens.flat[i] = 1 - self.sens.flat[i]
    
                # initialize the optimizer
                value = target_function()
                if value > value_best:
                    # save the state as the new best value
                    value_best, state_best = value, self.sens.copy()
                else:
                    # undo last change
                    self.sens.flat[i] = 1 - self.sens.flat[i]
                if ret_info:
                    info['values'].append(value)

        # sort the best state and store it in the current object
        state_best = self.sort_sensitivities(state_best)
        self.sens = state_best.copy()

        if ret_info:
            return value_best, state_best, info
        else:
            return value_best, state_best
        
    
    def optimize_library_anneal(self, target, steps, ret_info):
        """ optimizes the current library to maximize the result of the target
        function using simulated annealing. By default, the function returns the
        best value and the associated sensitivity matrix as result.        
        
        `steps` determines how many optimization steps we try
        `ret_info` determines whether extra information is returned from the
            optimization 
        """        
        # prepare the class that manages the simulated annealing
        annealer = ReceptorOptimizerAnnealer(self, target)
        annealer.steps = steps
        annealer.Tmax = self.parameters['anneal_Tmax']
        annealer.Tmin = self.parameters['anneal_Tmin']
        if self.parameters['verbosity'] == 0:
            annealer.updates = 0

        # do the optimization
        MI, state = annealer.optimize()

        # sort the best state and store it in the current object
        state = self.sort_sensitivities(state)
        self.sens = state.copy()
        
        if ret_info:
            return MI, state, annealer.info
        else:
            return MI, state    



def _ReceptorLibrary_mp_calc(args):
    """ helper function for multiprocessing """
    obj = ReceptorLibraryNumeric(**args[0])
    return getattr(obj, args[1])()

   
   
class ReceptorOptimizerAnnealer(Annealer):
    """ class that manages the simulated annealing """
    updates = 20    # Number of outputs
    copy_strategy = 'method'


    def __init__(self, model, target):
        """ initialize the optimizer with a `model` to run and a `target`
        function to call. """
        self.info = {}
        self.model = model
        self.target_func = getattr(model, target)
        super(ReceptorOptimizerAnnealer, self).__init__(model.sens)
   
   
    def move(self):
        """ change a single entry in the sensitivity matrix """   
        i = random.randrange(self.state.size)
        self.state.flat[i] = 1 - self.state.flat[i]

      
    def energy(self):
        """ returns the energy of the current state """
        self.model.sens = self.state
        MI = self.target_func()
        return -MI
    

    def optimize(self):
        """ optimizes the receptors and returns the best receptor set together
        with the achieved mutual information """
        state_best, energy_best = self.anneal()
        self.info['total_time'] = time.time() - self.start    
        self.info['states_considered'] = self.steps
        self.info['performance'] = self.steps / self.info['total_time']
        
        return -energy_best, state_best
   
   
   
def performance_test(Ns=15, Nr=3, frac=0.5):
    """ test the performance of the brute force and the monte carlo method """
    num = 2**Ns
    hs = np.random.random(Ns)
    model = ReceptorLibraryNumeric(Ns, Nr, hs, frac=frac)
    
    start = time.time()
    model.mutual_information_brute_force()
    time_brute_force = time.time() - start
    print('Brute force: %g sec' % time_brute_force)
    
    start = time.time()
    model.parameters['monte_carlo_strategy'] = 'frequency'
    model.mutual_information_monte_carlo(num)
    time_monte_carlo = time.time() - start
    print('Monte carlo, strat `frequency`: %g sec' % time_monte_carlo)
    
    start = time.time()
    model.parameters['monte_carlo_strategy'] = 'uniform'
    model.mutual_information_monte_carlo(num)
    time_monte_carlo = time.time() - start
    print('Monte carlo, strat `uniform`: %g sec' % time_monte_carlo)
    
    
            
if __name__ == '__main__':
    performance_test()
    

    
