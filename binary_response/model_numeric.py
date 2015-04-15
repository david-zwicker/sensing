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
import functools
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
        'max_num_receptors': 28,     #< prevents memory overflows
        'random_seed': None,         #< seed for the random number generator
        'interaction_matrix': None,  #< will be calculated if not given
        'inefficiency_weight': 1,    #< weighting parameter for inefficiency
        'brute_force_threshold_Ns': 10, #< largest Ns for using brute force 
        'monte_carlo_steps': 100000, #< default number of monte carlo steps
        'monte_carlo_strategy': 'frequency',
        'anneal_Tmax': 1e0,          #< Max (starting) temperature for annealing
        'anneal_Tmin': 1e-3,         #< Min (ending) temperature for annealing
        'verbosity': 0,              #< verbosity level    
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
        self.choose_interaction_matrix()


    @property
    def init_arguments(self):
        """ return the parameters of the model that can be used to reconstruct
        it by calling the __init__ method with these arguments """
        args = super(ReceptorLibraryNumeric, self).init_arguments
        args['frac'] = self.frac
        args['parameters'] = self.parameters
        return args


    @classmethod
    def get_random_arguments(cls):
        """ create random arguments for creating test instances """
        args = super(ReceptorLibraryNumeric, cls).get_random_arguments()
        frac = np.random.random()
        return args + [frac]


    def choose_interaction_matrix(self):
        """ creates a interaction matrix """
        shape = (self.Nr, self.Ns)
        # choose receptor substrate interaction randomly
        if self.parameters['interaction_matrix'] is None:
            self.int_mat = (np.random.random(shape) < self.frac).astype(np.int)
        else:
            self.int_mat = self.parameters['interaction_matrix']
            assert self.int_mat.shape == shape
            
            
    def sort_interaction_matrix(self, interaction_matrix=None):
        """ return the sorted `interaction_matrix` or sorts the internal
        interaction_matrix in place """
        if interaction_matrix is None:
            int_mat = self.int_mat
        else:
            int_mat = interaction_matrix

        data = [(sum(item), list(item)) for item in int_mat]
        int_mat = np.array([item[1] for item in sorted(data)])

        if interaction_matrix is None:
            self.int_mat = int_mat
        else:
            return int_mat
            

    def activity_single_brute_force(self):
        """ calculates the average activity of each receptor """
        prob_s = self.substrate_probability

        prob_a = np.zeros(self.Nr)
        # iterate over all mixtures
        for m in itertools.product((0, 1), repeat=self.Ns):
            # get the activity vector associated with m
            a = np.dot(self.int_mat, m).astype(np.bool)

            # probability of finding this substrate
            ma = np.array(m, np.bool)
            pm = np.prod(prob_s[ma]) * np.prod(1 - prob_s[~ma])
            prob_a[a] += pm
            
        return prob_a

            
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
            a = np.dot(self.int_mat, m).astype(np.bool)
            
            count_a[a] += 1
            
        return count_a/num


    def activity_correlations_brute_force(self):
        """ calculates the correlations between receptor activities """
        prob_s = self.substrate_probability

        prob_Caa = np.zeros((self.Nr, self.Nr))
        for m in itertools.product((0, 1), repeat=self.Ns):
            # get the associated output ...
            a = np.dot(self.int_mat, m).astype(np.bool)
            Caa = np.outer(a, a)

            # probability of finding this substrate
            ma = np.array(m, np.bool)
            pm = np.prod(prob_s[ma]) * np.prod(1 - prob_s[~ma])
            prob_Caa[Caa] += pm
        
        return prob_Caa
    
            
    def crosstalk(self):
        """ calculates the expected crosstalk between interaction matrices """
        return np.einsum('ai,bi,i->ab', self.int_mat, self.int_mat,
                         self.substrate_probability)


    def mutual_information(self, method='auto', ret_prob_activity=False):
        """ calculate the mutual information.
        
        `method` can be one of ['brute_force', 'monte_carlo', 'auto']. If 'auto'
            than the method is chosen automatically based on the problem size.
        `ret_prob_activity` determines whether the probabilities of the
            different outputs is returned or not
        """
        if method == 'auto':
            if self.Ns <= self.parameters['brute_force_threshold_Ns']:
                method = 'brute_force'
            else:
                method = 'monte_carlo'
                
        if method == 'brute_force':
            return self.mutual_information_brute_force(ret_prob_activity)
        elif method == 'monte_carlo':
            return self.mutual_information_monte_carlo(ret_prob_activity)
        elif method == 'estimate':
            if ret_prob_activity:
                raise NotImplementedError
            return self.mutual_information_estimate()
        else:
            raise ValueError('Unknown method `%s`.' % method)
            
            
    def mutual_information_brute_force(self, ret_prob_activity=False):
        """ calculate the mutual information by constructing all possible
        mixtures """
        base = 2 ** np.arange(self.Nr-1, -1, -1)

        prob_s = self.substrate_probability

        # prob_a contains the probability of finding activity a as an output.
        prob_a = np.zeros(2**self.Nr)
        for m in itertools.product((0, 1), repeat=self.Ns):
            # get the associated output ...
            a = np.dot(self.int_mat, m).astype(np.bool)
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
            
            
    def mutual_information_monte_carlo(self, ret_prob_activity=False):
        """ calculate the mutual information using a monte carlo strategy. The
        number of steps is given by the model parameter 'monte_carlo_steps' """
                
        base = 2 ** np.arange(self.Nr-1, -1, -1)
        prob_s = self.substrate_probability

        strategy = self.parameters['monte_carlo_strategy']
        steps = int(self.parameters['monte_carlo_steps'])
        if strategy == 'frequency':
            # sample mixtures according to the probabilities of finding
            # substrates
            count_a = np.zeros(2**self.Nr)
            for _ in xrange(steps):
                # choose a mixture vector according to substrate probabilities
                m = (np.random.random(self.Ns) < prob_s)
                
                # get the associated output ...
                a = np.dot(self.int_mat, m).astype(np.bool)
                # ... and represent it as a single integer
                a = np.dot(base, a)
                # increment counter for this output
                count_a[a] += 1
                
            # count_a contains the number of times output pattern a was observed.
            # We can thus construct P_a(a) from count_a. 
    
            prob_a = count_a / steps
            
        elif strategy == 'uniform':
            # sample mixtures with each substrate being equally likely and
            # correct the probabilities 
            prob_a = np.zeros(2**self.Nr)
            for _ in xrange(steps):
                # choose a mixture vector according to substrate probabilities
                m = np.random.randint(2, size=self.Ns)
                
                # get the associated output ...
                a = np.dot(self.int_mat, m).astype(np.bool)
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
    
        
    def mutual_information_estimate(self, approximate=False):
        """ returns a simple estimate of the mutual information """
        I_ai = self.int_mat
        prob_s = self.substrate_probability
        
        # calculate the probabilities of exciting receptors and pairs
        if approximate:
            # approximate calculation for small prob_s
            p_Ga = np.dot(I_ai, prob_s)
            p_Gab = np.einsum('ij,kj,j->ik', I_ai, I_ai, prob_s)
            assert np.all(p_Ga < 1) and np.all(p_Gab.flat < 1)
            
        else:
            # proper calculation of the cluster probabilities
            p_Ga = np.zeros(self.Nr)
            p_Gab = np.zeros((self.Nr, self.Nr))
            I_ai_mask = I_ai.astype(np.bool)
            for a in xrange(self.Nr):
                ps = prob_s[I_ai_mask[a, :]]
                p_Ga[a] = 1 - np.product(1 - ps)
                for b in xrange(a + 1, self.Nr):
                    ps = prob_s[I_ai_mask[a, :] * I_ai_mask[b, :]]
                    p_Gab[a, b] = 1 - np.product(1 - ps)
                    
        # calculate the approximate mutual information
        MI = self.Nr - 0.5*np.sum((1 - 2*p_Ga)**2)
        for a in xrange(self.Nr):
            for b in xrange(a + 1, self.Nr):
                MI -= 2*(1 - p_Ga[a] - p_Ga[b] + 3/4*p_Gab[a, b]) * p_Gab[a, b]
                
        return MI              
        
        
    def inefficiency_estimate(self):
        """ returns the estimated performance of the system, which acts as a
        proxy for the mutual information between input and output """
        I_ai = self.int_mat
        prob_s = self.substrate_probability
        
        # collect the terms describing the activity entropy
        term_entropy = np.sum((0.5 - np.prod(1 - I_ai*prob_s, axis=1)) ** 2)
        
        # collect the terms describing the crosstalk
        mat_ab = np.einsum('ij,kj,j->ik', I_ai, I_ai, prob_s)
        term_crosstalk = 2*np.sum(mat_ab[np.triu_indices(self.Nr, 1)]) 
        
        # add up the terms to produce the inefficiency parameter
        crosstalk_weight = self.parameters['inefficiency_weight']
        return term_entropy + crosstalk_weight*term_crosstalk
        
            
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
        
        
    def optimize_library(self, target, method='descent', direction='max',
                         steps=100, ret_info=False, args=None):
        """ optimizes the current library to maximize the result of the target
        function. By default, the function returns the best value and the
        associated interaction matrix as result.        
        
        `direction` is either 'min' or 'max' and determines whether a minimum
            or a maximum is sought.
        `steps` determines how many optimization steps we try 
        `ret_info` determines whether extra information is returned from the
            optimization 
        `args` is a dictionary of additional arguments that is passed to the
            target function

        `method` determines the method used for optimization. Supported are
            `descent`: simple gradient descent for `steps` number of steps
            `descent_parallel`: multiprocessing gradient descent. Note that this
                has an overhead and might actually decrease overall performance
                for small problems.
            `anneal`: simulated annealing
        """
        if method == 'descent':
            return self.optimize_library_descent(target, direction, steps,
                                                 multiprocessing=False,
                                                 ret_info=ret_info, args=args)
        elif method == 'descent_parallel':
            return self.optimize_library_descent(target, direction, steps,
                                                 multiprocessing=True,
                                                 ret_info=ret_info, args=args)
        elif method == 'anneal':
            return self.optimize_library_anneal(target, direction, steps,
                                                ret_info=ret_info, args=args)
            
        else:
            raise ValueError('Unknown optimization method `%s`' % method)
            
        
    def optimize_library_descent(self, target, direction='max', steps=100,
                                 multiprocessing=False, ret_info=False,
                                 args=None):
        """ optimizes the current library to maximize the result of the target
        function using gradient descent. By default, the function returns the
        best value and the associated interaction matrix as result.        
        
        `direction` is either 'min' or 'max' and determines whether a minimum
            or a maximum is sought.
        `steps` determines how many optimization steps we try
        `multiprocessing` is a flag deciding whether multiple processes are used
            to calculate the result. Note that this has an overhead and might
            actually decrease overall performance for small problems
        `ret_info` determines whether extra information is returned from the
            optimization 
        `args` is a dictionary of additional arguments that is passed to the
            target function
        """
        # get the target function to call
        target_function = getattr(self, target)
        if args is not None:
            target_function = functools.partial(target_function, **args)

        # initialize the optimizer
        value = target_function()
        value_best, state_best = value, self.int_mat.copy()
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
                    i = random.randrange(self.int_mat.size)
                    self.int_mat.flat[i] = 1 - self.int_mat.flat[i]
                    init_arguments['parameters']['interaction_matrix'] = self.int_mat
                    joblist.append((copy.deepcopy(init_arguments), target))
                    self.int_mat.flat[i] = 1 - self.int_mat.flat[i]
                    
                # run all the jobs
                results = pool.map(_ReceptorLibrary_mp_calc, joblist)
                
                # find the best result  
                if direction == 'max':              
                    res_best = np.argmax(results)
                    if results[res_best] > value_best:
                        value_best = results[res_best]
                        state_best = joblist[res_best][0]['parameters']['interaction_matrix']
                        # use the best state as a basis for the next iteration
                        self.int_mat = state_best
                        
                elif direction == 'min':
                    res_best = np.argmin(results)
                    if results[res_best] < value_best:
                        value_best = results[res_best]
                        state_best = joblist[res_best][0]['parameters']['interaction_matrix']
                        # use the best state as a basis for the next iteration
                        self.int_mat = state_best
                        
                else:
                    raise ValueError('Unsupported direction `%s`' % direction)
                        
                if ret_info:
                    info['values'].append(results[res_best])
                
        else:  
            # run the calculations in this process
            for _ in xrange(steps):
                # modify the current state
                i = random.randrange(self.int_mat.size)
                self.int_mat.flat[i] = 1 - self.int_mat.flat[i]
    
                # initialize the optimizer
                value = target_function()
                
                improved = ((direction == 'max' and value > value_best) or
                            (direction == 'min' and value < value_best))
                if improved:
                    # save the state as the new best value
                    value_best, state_best = value, self.int_mat.copy()
                else:
                    # undo last change
                    self.int_mat.flat[i] = 1 - self.int_mat.flat[i]
                    
                if ret_info:
                    info['values'].append(value)

        # sort the best state and store it in the current object
        state_best = self.sort_interaction_matrix(state_best)
        self.int_mat = state_best.copy()

        if ret_info:
            return value_best, state_best, info
        else:
            return value_best, state_best
        
    
    def optimize_library_anneal(self, target, direction='max', steps=100,
                                ret_info=False, args=None):
        """ optimizes the current library to maximize the result of the target
        function using simulated annealing. By default, the function returns the
        best value and the associated interaction matrix as result.        
        
        `direction` is either 'min' or 'max' and determines whether a minimum
            or a maximum is sought.
        `steps` determines how many optimization steps we try
        `ret_info` determines whether extra information is returned from the
            optimization 
        `args` is a dictionary of additional arguments that is passed to the
            target function
        """        
        # prepare the class that manages the simulated annealing
        annealer = ReceptorOptimizerAnnealer(self, target, direction, args)
        annealer.steps = steps
        annealer.Tmax = self.parameters['anneal_Tmax']
        annealer.Tmin = self.parameters['anneal_Tmin']
        if self.parameters['verbosity'] == 0:
            annealer.updates = 0

        # do the optimization
        MI, state = annealer.optimize()

        # sort the best state and store it in the current object
        state = self.sort_interaction_matrix(state)
        self.int_mat = state.copy()
        
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


    def __init__(self, model, target, direction='max', args=None):
        """ initialize the optimizer with a `model` to run and a `target`
        function to call. """
        self.info = {}
        self.model = model

        target_function = getattr(model, target)
        if args is not None:
            self.target_func = functools.partial(target_function, **args)
        else:
            self.target_func = target_function

        self.direction = direction
        super(ReceptorOptimizerAnnealer, self).__init__(model.int_mat)
   
   
    def move(self):
        """ change a single entry in the interaction matrix """   
        i = random.randrange(self.state.size)
        self.state.flat[i] = 1 - self.state.flat[i]

      
    def energy(self):
        """ returns the energy of the current state """
        self.model.int_mat = self.state
        value = self.target_func()
        if self.direction == 'max':
            return -value
        else:
            return value
    

    def optimize(self):
        """ optimizes the receptors and returns the best receptor set together
        with the achieved mutual information """
        state_best, value_best = self.anneal()
        self.info['total_time'] = time.time() - self.start    
        self.info['states_considered'] = self.steps
        self.info['performance'] = self.steps / self.info['total_time']
        
        if self.direction == 'max':
            return -value_best, state_best
        else:
            return value_best, state_best
   
   
   
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
    print('Monte carlo, strategy `frequency`: %g sec' % time_monte_carlo)
    
    start = time.time()
    model.parameters['monte_carlo_strategy'] = 'uniform'
    model.mutual_information_monte_carlo(num)
    time_monte_carlo = time.time() - start
    print('Monte carlo, strategy `uniform`: %g sec' % time_monte_carlo)
    
    
            
if __name__ == '__main__':
    performance_test()
    

    
