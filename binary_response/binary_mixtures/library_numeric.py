'''
Created on Mar 27, 2015

@author: David Zwicker <dzwicker@seas.harvard.edu>

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
from six.moves import range, zip

from .library_base import LibraryBinaryBase



class LibraryBinaryNumeric(LibraryBinaryBase):
    """ represents a single receptor library that handles binary mixtures """
    
    # default parameters that are used to initialize a class if not overwritten
    parameters_default = {
        'max_num_receptors': 28,    #< prevents memory overflows
        'interaction_matrix': None, #< will be calculated if not given
        'interaction_matrix_params': None, #< parameters determining I_ai
        'inefficiency_weight': 1,   #< weighting parameter for inefficiency
        'brute_force_threshold_Ns': 10, #< largest Ns for using brute force 
        'monte_carlo_steps': 1e5,   #< default number of Monte Carlo steps
        'metropolis_steps': 1e5,    #< default number of Metropolis steps
        'anneal_Tmax': 1e0,         #< Max (starting) temperature for annealing
        'anneal_Tmin': 1e-3,        #< Min (ending) temperature for annealing
        'verbosity': 0,             #< verbosity level    
    }
    

    def __init__(self, num_substrates, num_receptors, parameters=None):
        """ initialize a receptor library by setting the number of receptors,
        the number of substrates it can respond to, and optional additional
        parameters in the parameter dictionary """
        # the call to the inherited method also sets the default parameters from
        # this class
        super(LibraryBinaryNumeric, self).__init__(num_substrates,
                                                   num_receptors, parameters)        

        # prevent integer overflow in collecting activity patterns
        assert num_receptors <= self.parameters['max_num_receptors'] <= 63

        int_mat_shape = (self.Nr, self.Ns)
        if self.parameters['interaction_matrix'] is not None:
            # copy the given matrix
            self.int_mat[:] = self.parameters['interaction_matrix']
            assert self.int_mat.shape == int_mat_shape
        elif self.parameters['interaction_matrix_params'] is not None:
            # create a matrix with the given properties
            params = self.parameters['interaction_matrix_params']
            self.choose_interaction_matrix(**params)
        else:
            # initialize the interaction matrix with zeros
            self.int_mat = np.zeros(int_mat_shape, np.uint8)


    @classmethod
    def create_test_instance(cls, **kwargs):
        """ creates a test instance used for consistency tests """
        obj = super(LibraryBinaryNumeric, cls).create_test_instance(**kwargs)
        # determine optimal parameters for the interaction matrix
        from .library_theory import LibraryBinaryUniform
        theory = LibraryBinaryUniform.from_other(obj)
        obj.choose_interaction_matrix(**theory.get_optimal_library())
        return obj
    

    def choose_interaction_matrix(self, density=0, avoid_correlations=False):
        """ creates a interaction matrix with the given properties """
        shape = (self.Nr, self.Ns)
        if density == 0:
            # simple case of empty matrix
            self.int_mat = np.zeros(shape, np.uint8)
        elif density >= 1:
            # simple case of full matrix
            self.int_mat = np.ones(shape, np.uint8)
            
        elif avoid_correlations:
            # choose receptor substrate interaction randomly but try to avoid
            # correlations between the receptors
            self.int_mat = np.zeros(shape, np.uint8)
            num_entries = int(round(density * self.Nr * self.Ns))
            
            empty_int_mat = True
            while num_entries > 0:
                # specify the substrates that we want to detect
                if num_entries >= self.Ns:
                    i_ids = np.arange(self.Ns)
                    num_entries -= self.Ns
                else:
                    i_ids = np.random.choice(np.arange(self.Ns), num_entries,
                                             replace=False)
                    num_entries = 0
                    
                if empty_int_mat:
                    # set the receptors for the substrates
                    a_ids = np.random.randint(0, self.Nr, len(i_ids))
                    for i, a in zip(i_ids, a_ids):
                        self.int_mat[a, i] = 1
                    empty_int_mat = False
                    
                else:
                    # choose receptors for each substrate from the ones that
                    # are not activated, yet
                    for i in i_ids:
                        a_ids = np.flatnonzero(self.int_mat[:, i] == 0)
                        self.int_mat[random.choice(a_ids), i] = 1
            
        else: # not avoid_correlations:
            # choose receptor substrate interaction randomly and don't worry
            # about correlations
            self.int_mat = (np.random.random(shape) < density).astype(np.uint8)
            
        # save the parameters determining this matrix
        self.parameters['interaction_matrix_params'] = {
            'density': density,
            'avoid_correlations': avoid_correlations
        }   
            
            
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
            
            
    def mixture_statistics(self):
        """ calculates statistics of mixtures. Returns a vector with the 
        frequencies at which substrates are present in mixtures and a matrix
        of correlations among substrates """
        
        if self.has_correlations:
            # mixture has correlations => we do Metropolis sampling
            if self.Ns <= self.parameters['brute_force_threshold_Ns']:
                ci_mean, cij_corr = self.mixture_statistics_brute_force()
            else:
                ci_mean, cij_corr = self.mixture_statistics_metropolis()
                
        else:
            # mixture does not have correlations => we can calculated the
            # statistics directly
            ci_mean = self.substrate_probabilities
            cij_corr = np.diag(ci_mean - ci_mean**2)
            
        return ci_mean, cij_corr
    
    
    def mixture_statistics_brute_force(self):
        """ calculates mixture statistics using a brute force algorithm """
        hi = self.commonness
        Jij = self.correlations
        
        Z = 0
        hist1d = np.zeros(self.Ns)
        hist2d = np.zeros((self.Ns, self.Ns))
        
        # iterate over all mixtures
        for c in itertools.product((0, 1), repeat=self.Ns):
            c = np.array(c)
            
            # probability of finding this mixture
            prob_c = np.exp(np.dot(np.dot(Jij, c) + hi, c))
            Z += prob_c        
            hist1d += c * prob_c
            hist2d += np.outer(c, c) * prob_c
        
        # calculate the frequency and the correlations 
        ci_mean = hist1d / Z
        cij = hist2d / Z
        cij_corr = cij - np.outer(ci_mean, ci_mean)
        
        return ci_mean, cij_corr  
    
    
    def mixture_statistics_metropolis(self):
        """ calculates mixture statistics using a metropolis algorithm """
        hi = self.commonness
        Jij = self.correlations
        
        c = np.random.random_integers(0, 1, self.Ns)
        Elast = -np.dot(np.dot(Jij, c) + hi, c)
        
        count = 0
        hist1d = np.zeros(self.Ns, np.int)
        hist2d = np.zeros((self.Ns, self.Ns), np.int)
        for _ in range(int(self.parameters['metropolis_steps'])):
            i = random.randrange(self.Ns)
            c[i] = 1 - c[i]
            Ei = -np.dot(np.dot(Jij, c) + hi, c)
            if Ei < Elast or random.random() < np.exp(Elast - Ei):
                # accept the new state
                Elast = Ei
            else:
                # reject the new state and revert to the last one
                c[i] = 1 - c[i]
        
            # accept the state
            count += 1
            hist1d += c
            hist2d += np.outer(c, c)
        
        # calculate the frequency and the correlations 
        ci_mean = hist1d / count
        cij = hist2d / count
        cij_corr = cij - np.outer(ci_mean, ci_mean)
        
        return ci_mean, cij_corr
            

    def activity_single(self, method='auto'):
        """ calculates the average activity of each receptor
        
        `method` can be one of ['brute_force', 'monte_carlo', 'auto']. If 'auto'
            than the method is chosen automatically based on the problem size.
        """
        if method == 'auto':
            if self.Ns <= self.parameters['brute_force_threshold_Ns']:
                method = 'brute_force'
            elif self.has_correlations:
                method = 'metropolis'
            else:
                method = 'monte_carlo'
                
        if method == 'brute_force':
            return self.activity_single_brute_force()
        elif method == 'monte_carlo':
            return self.activity_single_monte_carlo()
        elif method == 'metropolis':
            return self.activity_single_metropolis()
        elif method == 'estimate':
            return self.activity_single_estimate()
        else:
            raise ValueError('Unknown method `%s`.' % method)
        
        
    def activity_single_brute_force(self):
        """ calculates the average activity of each receptor """
        hi = self.commonness
        Jij = self.correlations

        prob_a = np.zeros(self.Nr)
        Z = 0
        
        # iterate over all mixtures
        for c in itertools.product((0, 1), repeat=self.Ns):
            c = np.array(c)
            
            # get the activity vector associated with m
            a = np.dot(self.int_mat, c).astype(np.bool)

            # probability of finding this mixture
            prob_c = np.exp(np.dot(np.dot(Jij, c) + hi, c))
            prob_a[a] += prob_c
            Z += prob_c
                
        return prob_a / Z

            
    def activity_single_monte_carlo(self, num=None):
        """ calculates the average activity of each receptor """ 
        if self.has_correlations:
            raise NotImplementedError('Not implemented for correlated mixtures')

        if num is None:
            num = int(self.parameters['monte_carlo_steps'])        
    
        prob_s = self.substrate_probabilities
    
        count_a = np.zeros(self.Nr)
        for _ in range(num):
            # choose a mixture vector according to substrate probabilities
            c = (np.random.random(self.Ns) < prob_s)
            
            # get the associated output
            a = np.dot(self.int_mat, c).astype(np.bool)
            
            count_a[a] += 1
            
        # return the normalized output
        return count_a/num
    
    
    def activity_single_metropolis(self, num=None):
        """ calculates the average activity of each receptor """ 
        if num is None:
            num = int(self.parameters['metropolis_steps'])        
    
        hi = self.commonness
        Jij = self.correlations
        
        # start with a random concentration vector 
        c = np.random.random_integers(0, 1, self.Ns)
        Elast = -np.dot(np.dot(Jij, c) + hi, c)
        
        count_a = np.zeros(self.Nr)
        for _ in range(num):
            i = random.randrange(self.Ns)
            c[i] = 1 - c[i]
            Ei = -np.dot(np.dot(Jij, c) + hi, c)
            if Ei < Elast or random.random() < np.exp(Elast - Ei):
                # accept the new state
                Elast = Ei
            else:
                # reject the new state and revert to the last one
                c[i] = 1 - c[i]
        
            # accept the state and get the associated output
            a = np.dot(self.int_mat, c).astype(np.bool)
            count_a[a] += 1

        # return the normalized output
        return count_a / num
            
    
    def activity_single_estimate(self, approx_prob=False):
        """ estimates the average activity of each receptor. 
        `approx_prob` determines whether the probabilities of encountering
            substrates in mixtures are calculated exactly or only approximative,
            which should work for small probabilities. """
        if self.has_correlations:
            raise NotImplementedError('Not implemented for correlated mixtures')

        I_ai = self.int_mat
        prob_s = self.substrate_probabilities

        # calculate the probabilities of exciting receptors and pairs
        if approx_prob:
            # approximate calculation for small prob_s
            p_Ga = np.dot(I_ai, prob_s)
            assert np.all(p_Ga < 1)
            
        else:
            # proper calculation of the cluster probabilities
            p_Ga = np.zeros(self.Nr)
            I_ai_mask = I_ai.astype(np.bool)
            for a in range(self.Nr):
                p_Ga[a] = 1 - np.product(1 - prob_s[I_ai_mask[a, :]])
        return p_Ga


    def activity_correlations_brute_force(self):
        """ calculates the correlations between receptor activities """
        if self.has_correlations:
            raise NotImplementedError('Not implemented for correlated mixtures')

        prob_s = self.substrate_probabilities

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
                         self.substrate_probabilities)


    def mutual_information(self, method='auto', **kwargs):
        """ calculate the mutual information.
        
        `method` can be one of ['brute_force', 'monte_carlo', 'auto']. If 'auto'
            than the method is chosen automatically based on the problem size.
        `ret_prob_activity` determines whether the probabilities of the
            different outputs is returned or not
        """
        if method == 'auto':
            if self.Ns <= self.parameters['brute_force_threshold_Ns']:
                method = 'brute_force'
            elif self.has_correlations:
                method = 'metropolis'
            else:
                method = 'monte_carlo'
                
        if method == 'brute_force':
            return self.mutual_information_brute_force(**kwargs)
        elif method == 'monte_carlo':
            return self.mutual_information_monte_carlo(**kwargs)
        elif method == 'metropolis':
            return self.mutual_information_metropolis(**kwargs)
        elif method == 'estimate':
            return self.mutual_information_estimate(**kwargs)
        else:
            raise ValueError('Unknown method `%s`.' % method)
            
            
    def mutual_information_brute_force(self, ret_prob_activity=False):
        """ calculate the mutual information by constructing all possible
        mixtures """
        hi = self.commonness
        Jij = self.correlations
        base = 2 ** np.arange(0, self.Nr)

        # prob_a contains the probability of finding activity a as an output.
        prob_a = np.zeros(2**self.Nr)
        for c in itertools.product((0, 1), repeat=self.Ns):
            c = np.array(c)

            # probability of finding this mixture
            prob_c = np.exp(np.dot(np.dot(Jij, c) + hi, c))
            
            # get the associated output ...
            a = np.dot(self.int_mat, c).astype(np.bool)
            # ... and represent it as a single integer
            a = np.dot(base, a)

            prob_a[a] += prob_c
            
        # normalize the output to make it a probability distribution
        prob_a /= prob_a.sum()
        
        # calculate the mutual information
        MI = -sum(pa*np.log2(pa) for pa in prob_a if pa != 0)
        
        if ret_prob_activity:
            return MI, prob_a.mean()
        else:
            return MI
            
            
    def mutual_information_monte_carlo(self, ret_error=False,
                                       ret_prob_activity=False):
        """ calculate the mutual information using a Monte Carlo strategy. The
        number of steps is given by the model parameter 'monte_carlo_steps' """
        if self.has_correlations:
            raise NotImplementedError('Not implemented for correlated mixtures')
                
        base = 2 ** np.arange(0, self.Nr)
        prob_s = self.substrate_probabilities

        steps = int(self.parameters['monte_carlo_steps'])

        # sample mixtures according to the probabilities of finding
        # substrates
        count_a = np.zeros(2**self.Nr)
        for _ in range(steps):
            # choose a mixture vector according to substrate probabilities
            c = (np.random.random(self.Ns) < prob_s)
            
            # get the associated output ...
            a = np.dot(self.int_mat, c).astype(np.bool)
            # ... and represent it as a single integer
            a = np.dot(base, a)
            # increment counter for this output
            count_a[a] += 1
        
        # count_a contains the number of times output pattern a was observed.
        # We can thus construct P_a(a) from count_a. 
        prob_a = count_a / steps
        # count_a_err = prob_a * np.sqrt(steps)
        # prob_a_err = count_a_err / steps = prob_a / np.sqrt(steps) / steps
        
        # calculate the mutual information from the result pattern
        MI = -sum(pa*np.log2(pa) for pa in prob_a if pa != 0)
        if ret_error:
            # estimate the error of the mutual information calculation
            MI_err = sum(np.abs(1/np.log(2) + np.log2(pa)) * pa
                         for pa in prob_a if pa != 0) / np.sqrt(steps)

            if ret_prob_activity:
                return MI, MI_err, prob_a
            else:
                return MI, MI_err

        else:
            # error should not be calculated       
            if ret_prob_activity:
                return MI, prob_a
            else:
                return MI
        
        
    def mutual_information_monte_carlo_extrapolate(self, ret_prob_activity=False):
        """ calculate the mutual information using a Monte Carlo strategy. The
        number of steps is given by the model parameter 'monte_carlo_steps' """
        if self.has_correlations:
            raise NotImplementedError('Not implemented for correlated mixtures')
                
        base = 2 ** np.arange(0, self.Nr)
        prob_s = self.substrate_probabilities

        max_steps = int(self.parameters['monte_carlo_steps'])
        steps, MIs = [], []

        # sample mixtures according to the probabilities of finding
        # substrates
        count_a = np.zeros(2**self.Nr)
        step_check = 10000
        for step in range(max_steps):
            # choose a mixture vector according to substrate probabilities
            m = (np.random.random(self.Ns) < prob_s)
            
            # get the associated output ...
            a = np.dot(self.int_mat, m).astype(np.bool)
            # ... and represent it as a single integer
            a = np.dot(base, a)
            # increment counter for this output
            count_a[a] += 1

            if step == step_check - 1:
                # do an extrapolation step
                # calculate the mutual information from the result pattern
                prob_a = count_a / step
                MI = -sum(pa*np.log2(pa) for pa in prob_a if pa != 0)
                
                # save the data                
                steps.append(step)
                MIs.append(MI)
                
                # do the extrapolation
                if len(steps) >= 3:
                    a2, a1, a0 = MIs[-3:]
                    MI_ext = (a0*a2 - a1*a1)/(a0 - 2*a1 + a2)
#                     MI_ext = self._get_extrapolated_mutual_information(steps, MIs)
                    print((step, MIs[-1], MI_ext))
                    
                step_check += 10000
            
        else:
            # count_a contains the number of times output pattern a was observed.
            # We can thus construct P_a(a) from count_a. 
            
            # calculate the mutual information from the result pattern
            prob_a = count_a / step
            MI = -sum(pa*np.log2(pa) for pa in prob_a if pa != 0)

        if ret_prob_activity:
            return MI, prob_a
        else:
            return MI
                

    def mutual_information_metropolis(self, ret_prob_activity=False):
        """ calculates the average activity of each receptor """ 
    
        hi = self.commonness
        Jij = self.correlations
        base = 2 ** np.arange(0, self.Nr)
        
        # start with a random concentration vector 
        c = np.random.random_integers(0, 1, self.Ns)
        Elast = -np.dot(np.dot(Jij, c) + hi, c)
        
        steps = int(self.parameters['metropolis_steps'])        
        count_a = np.zeros(2**self.Nr)
        for _ in range(steps):
            i = random.randrange(self.Ns)
            c[i] = 1 - c[i]
            Ei = -np.dot(np.dot(Jij, c) + hi, c)
            if Ei < Elast or random.random() < np.exp(Elast - Ei):
                # accept the new state
                Elast = Ei
            else:
                # reject the new state and revert to the last one
                c[i] = 1 - c[i]

            # accept the state, get the associated output ...
            a = np.dot(self.int_mat, c).astype(np.bool)
            # ... and represent it as a single integer
            a = np.dot(base, a)
            # increment counter for this output
            count_a[a] += 1

        # count_a contains the number of times output pattern a was observed.
        # We can thus construct P_a(a) from count_a. 
        prob_a = count_a / steps
        
        # calculate the mutual information from the result pattern
        MI = -sum(pa*np.log2(pa) for pa in prob_a if pa != 0)
        # error should not be calculated       
        if ret_prob_activity:
            return MI, prob_a
        else:
            return MI        
            
                    
    def mutual_information_estimate(self, approx_prob=False):
        """ returns a simple estimate of the mutual information.
        `approx_prob` determines whether the probabilities of encountering
            substrates in mixtures are calculated exactly or only approximative,
            which should work for small probabilities. """
        if self.has_correlations:
            raise NotImplementedError('Not implemented for correlated mixtures')

        I_ai = self.int_mat
        prob_s = self.substrate_probabilities
        
        # calculate the probabilities of exciting receptors and pairs
        if approx_prob:
            # approximate calculation for small prob_s
            p_Ga = np.dot(I_ai, prob_s)
            p_Gab = np.einsum('ij,kj,j->ik', I_ai, I_ai, prob_s)
            assert np.all(p_Ga < 1) and np.all(p_Gab.flat < 1)
            
        else:
            # proper calculation of the cluster probabilities
            p_Ga = np.zeros(self.Nr)
            p_Gab = np.zeros((self.Nr, self.Nr))
            I_ai_mask = I_ai.astype(np.bool)
            for a in range(self.Nr):
                ps = prob_s[I_ai_mask[a, :]]
                p_Ga[a] = 1 - np.product(1 - ps)
                for b in range(a + 1, self.Nr):
                    ps = prob_s[I_ai_mask[a, :] * I_ai_mask[b, :]]
                    p_Gab[a, b] = 1 - np.product(1 - ps)
                    
        # calculate the approximate mutual information
        MI = self.Nr - 0.5*np.sum((1 - 2*p_Ga)**2)
        for a in range(self.Nr):
            for b in range(a + 1, self.Nr):
                MI -= 2*(1 - p_Ga[a] - p_Ga[b] + 3/4*p_Gab[a, b]) * p_Gab[a, b]
                
        return MI              
        
        
    def inefficiency_estimate(self):
        """ returns the estimated performance of the system, which acts as a
        proxy for the mutual information between input and output """
        if self.has_correlations:
            raise NotImplementedError('Not implemented for correlated mixtures')

        I_ai = self.int_mat
        prob_s = self.substrate_probabilities
        
        # collect the terms describing the activity entropy
        term_entropy = np.sum((0.5 - np.prod(1 - I_ai*prob_s, axis=1)) ** 2)
        
        # collect the terms describing the crosstalk
        mat_ab = np.einsum('ij,kj,j->ik', I_ai, I_ai, prob_s)
        term_crosstalk = 2*np.sum(mat_ab[np.triu_indices(self.Nr, 1)]) 
        
        # add up the terms to produce the inefficiency parameter
        crosstalk_weight = self.parameters['inefficiency_weight']
        return term_entropy + crosstalk_weight*term_crosstalk
        
        
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
            for _ in range(steps // pool_size):
                joblist = []
                init_arguments = self.init_arguments
                for _ in range(pool_size):
                    # modify the current state and add it to the job list
                    i = random.randrange(self.int_mat.size)
                    self.int_mat.flat[i] = 1 - self.int_mat.flat[i]
                    init_arguments['parameters']['interaction_matrix'] = self.int_mat
                    joblist.append((copy.deepcopy(init_arguments), target))
                    self.int_mat.flat[i] = 1 - self.int_mat.flat[i]
                    
                # run all the jobs
                results = pool.map(_optimize_library_job, joblist)
                
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
            for _ in range(steps):
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
        # lazy import
        from .optimizer import ReceptorOptimizerAnnealer  # @UnresolvedImport
        
        # prepare the class that manages the simulated annealing
        annealer = ReceptorOptimizerAnnealer(self, target, direction, args)
        annealer.steps = int(steps)
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



def _optimize_library_job(args):
    """ helper function for optimizing the receptor library using
    multiprocessing """
    # Note that we do not set the seed of the random number generator because
    # we already modified the interaction matrix before calling this function
    # and it does not harm us when all sub processes have the same sequence of
    # random numbers.
    
    # create the object ...
    obj = LibraryBinaryNumeric(**args[0])
    # ... and evaluate the requested method 
    return getattr(obj, args[1])()

   
   
def performance_test(Ns=15, Nr=3):
    """ test the performance of the brute force and the Monte Carlo method """
    num = 2**Ns
    hs = np.random.random(Ns)
    model = LibraryBinaryNumeric(Ns, Nr, hs)
    
    start = time.time()
    model.mutual_information_brute_force()
    time_brute_force = time.time() - start
    print('Brute force: %g sec' % time_brute_force)
    
    start = time.time()
    model.mutual_information_monte_carlo(num)
    time_monte_carlo = time.time() - start
    print('Monte carlo: %g sec' % time_monte_carlo)
    
        
            
if __name__ == '__main__':
    performance_test()
    

    
