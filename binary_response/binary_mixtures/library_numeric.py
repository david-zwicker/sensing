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
import collections
import functools
import itertools
import time
import multiprocessing as mp
import random

import numpy as np
import scipy.misc
from six.moves import range, zip

from .library_base import LibraryBinaryBase



LN2 = np.log(2)



class LibraryBinaryNumeric(LibraryBinaryBase):
    """ represents a single receptor library that handles binary mixtures """
    
    # default parameters that are used to initialize a class if not overwritten
    parameters_default = {
        'max_num_receptors': 28,        #< prevents memory overflows
        'max_steps': 1e7,               #< maximal number of steps 
        'interaction_matrix': None,     #< will be calculated if not given
        'interaction_matrix_params': None, #< parameters determining S_ni
        'inefficiency_weight': 1,       #< weighting parameter for inefficiency
        'brute_force_threshold_Ns': 10, #< largest Ns for using brute force 
        'monte_carlo_steps': 'auto',    #< default steps for monte carlo
        'monte_carlo_steps_min': 1e4,   #< minimal steps for monte carlo
        'monte_carlo_steps_max': 1e5,   #< maximal steps for monte carlo
        'metropolis_steps': 1e5,        #< default number of Metropolis steps
        'metropolis_steps_min': 1e4,    #< minimal steps for metropolis
        'metropolis_steps_max': 1e5,    #< maximal steps for metropolis
        'fixed_mixture_size': None,     #< fixed m or None
        'optimizer_values_count': 1024, #< maximal number of values stored 
        'anneal_Tmax': 1e-1,            #< Max (starting) temp. for annealing
        'anneal_Tmin': 1e-3,            #< Min (ending) temp. for annealing
        'verbosity': 0,                 #< verbosity level    
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

        initialize_state = self.parameters['initialize_state'] 
        int_mat_shape = (self.Nr, self.Ns)
        
        if initialize_state is None:
            # do not initialize with anything
            self.int_mat = np.zeros(int_mat_shape, np.uint8)
            
        elif initialize_state == 'exact':
            # initialize the state using saved parameters
                self.int_mat = self.parameters['interaction_matrix'].copy()
            
        elif initialize_state == 'ensemble':
            # initialize the state using the ensemble parameters
                params = self.parameters['interaction_matrix_params']
                self.choose_interaction_matrix(**params)
            
        elif initialize_state == 'auto':
            # use exact values if saved or ensemble properties otherwise
            if self.parameters['interaction_matrix'] is not None:
                # copy the given matrix
                self.int_mat = self.parameters['interaction_matrix'].copy()
            elif self.parameters['interaction_matrix_params'] is not None:
                # create a matrix with the given properties
                params = self.parameters['interaction_matrix_params']
                self.choose_interaction_matrix(**params)
            else:
                # initialize the interaction matrix with zeros
                self.int_mat = np.zeros(int_mat_shape, np.uint8)

        else:
            raise ValueError('Unknown initialization protocol `%s`' % 
                             initialize_state)

        assert self.int_mat.shape == int_mat_shape


    @classmethod
    def get_random_arguments(cls, **kwargs):
        """ create random arguments for creating test instances """
        args = super(LibraryBinaryNumeric, cls).get_random_arguments(**kwargs)
        if 'fixed_mixture_size' in kwargs:
            args['parameters']['fixed_mixture_size'] = \
                                                kwargs['fixed_mixture_size']
        return args


    @classmethod
    def create_test_instance(cls, **kwargs):
        """ creates a test instance used for consistency tests """
        # create a instance with random parameters
        obj = super(LibraryBinaryNumeric, cls).create_test_instance(**kwargs)
        # choose an optimal interaction matrix
        obj.choose_interaction_matrix('auto')
        return obj
    
                
    def get_steps(self, scheme):
        """ calculate the number of steps to do for `scheme`"""
        if scheme == 'monte_carlo':
            # calculate the number of steps for a monte-carlo scheme
            if self.parameters['monte_carlo_steps'] == 'auto':
                steps_min = self.parameters['monte_carlo_steps_min']
                steps_max = self.parameters['monte_carlo_steps_max']
                steps = np.clip(10 * 2**self.Nr, steps_min, steps_max) 
                # Here, the factor 10 is an arbitrary scaling factor
            else:
                steps = self.parameters['monte_carlo_steps']
            
        elif scheme == 'metropolis':
            # calculate the number of steps for a metropolis scheme
            if self.parameters['metropolis_steps'] == 'auto':
                steps_min = self.parameters['metropolis_steps_min']
                steps_max = self.parameters['metropolis_steps_max']
                steps = np.clip(10 * 2**self.Nr, steps_min, steps_max) 
                # Here, the factor 10 is an arbitrary scaling factor
            else:
                steps = self.parameters['metropolis_steps']
                
        else:
            raise ValueError('Unknown stepping scheme `%s`' % scheme)
            
        return int(steps)


    def choose_interaction_matrix(self, density=0, avoid_correlations=False):
        """ creates a interaction matrix with the given properties """
        shape = (self.Nr, self.Ns)
        
        if density == 'auto':
            # determine optimal parameters for the interaction matrix
            from .library_theory import LibraryBinaryUniform
            theory = LibraryBinaryUniform.from_other(self)
            density = theory.get_optimal_library()['density']
            
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
        interaction_matrix in place. This function rearranges receptors such
        that receptors reacting to an equal number of substrates and to similar
        substrates are close together. """
        
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
        
        
    @property
    def _iterate_steps(self):
        """ return the number of steps we iterate over """
        mixture_size = self.parameters['fixed_mixture_size']
        if mixture_size is None:
            return 2 ** self.Ns
        else:
            return scipy.misc.comb(self.Ns, mixture_size, exact=True)
        
        
    def _iterate_mixtures(self):
        """ iterate over all mixtures and yield the mixture with probability """
        
        if self._iterate_steps > self.parameters['max_steps']:
            raise RuntimeError('The iteration would take more than %g steps'
                               % self.parameters['max_steps'])
        
        hi = self.commonness
        Jij = self.correlations

        mixture_size = self.parameters['fixed_mixture_size']
        if mixture_size is None:
            # iterate over all mixtures
            for c in itertools.product((0, 1), repeat=self.Ns):
                c =  np.array(c, np.uint8)
                weight_c = np.exp(np.dot(np.dot(Jij, c) + hi, c))
                yield c, weight_c
                
        else:
            # iterate over all mixtures with constant number of substrates
            c = np.zeros(self.Ns, np.uint8)
            for nz in itertools.combinations(range(self.Ns), mixture_size):
                c[:] = 0
                c[np.array(nz)] = 1
                weight_c = np.exp(np.dot(np.dot(Jij, c) + hi, c))
                yield c, weight_c


    @property
    def _sample_steps(self):
        """ returns the number of steps that are sampled """
        mixture_size = self.parameters['fixed_mixture_size']
        if not self.has_correlations and mixture_size is None:
            return self.get_steps('monte_carlo')
        else:
            return self.get_steps('metropolis')


    def _sample_mixtures(self):
        """ sample mixtures with uniform probability yielding single mixtures """
                
        mixture_size = self.parameters['fixed_mixture_size']
                
        if not self.has_correlations and mixture_size is None:
            # use simple monte carlo algorithm
            prob_s = self.substrate_probabilities
            
            for _ in range(self._sample_steps):
                # choose a mixture vector according to substrate probabilities
                yield (np.random.random(self.Ns) < prob_s)
                
        else:            
            # use metropolis algorithm
            hi = self.commonness
            Jij = self.correlations

            if mixture_size is None:
                # go through all mixtures and don't keep the size constant
                
                # start with a random concentration vector 
                c = np.random.random_integers(0, 1, self.Ns)
                E_last = -np.dot(np.dot(Jij, c) + hi, c)
                
                for _ in range(self._sample_steps):
                    i = random.randrange(self.Ns)
                    c[i] = 1 - c[i] #< switch the entry
                    Ei = -np.dot(np.dot(Jij, c) + hi, c)
                    if Ei < E_last or random.random() < np.exp(E_last - Ei):
                        # accept the new state
                        E_last = Ei
                    else:
                        # reject the new state and revert to the last one
                        c[i] = 1 - c[i]
                
                    yield c
                            
            else:
                # go through mixtures with keeping their size constant

                # create random concentration vector with fixed substrate count
                c = np.r_[np.ones(mixture_size, np.uint),
                          np.zeros(self.Ns - mixture_size, np.uint)]
                np.random.shuffle(c)
                E_last = -np.dot(np.dot(Jij, c) + hi, c)
                
                for _ in range(self._sample_steps):
                    # find the next mixture by swapping two items
                    i0 = random.choice(np.flatnonzero(c == 0)) #< find 0
                    i1 = random.choice(np.flatnonzero(c))      #< find 1
                    c[i0], c[i1] = 1, 0 #< swap entries
                    Ei = -np.dot(np.dot(Jij, c) + hi, c)
                    if Ei < E_last or random.random() < np.exp(E_last - Ei):
                        # accept the new state
                        E_last = Ei
                    else:
                        # reject the new state and revert to the last one
                        c[i0], c[i1] = 0, 1
                
                    yield c
                        
            
    def mixture_statistics(self):
        """ calculates statistics of mixtures. Returns a vector with the 
        frequencies at which substrates are present in mixtures and a matrix
        of correlations among substrates """
        
        fixed_mixture_size = self.parameters['fixed_mixture_size']
        
        if self.has_correlations or fixed_mixture_size is not None:
            # mixture has correlations => we do Metropolis sampling
            if self.Ns <= self.parameters['brute_force_threshold_Ns']:
                ci_mean, cij_corr = self.mixture_statistics_brute_force()
            else:
                ci_mean, cij_corr = self.mixture_statistics_monte_carlo()
                
        else:
            # mixture does not have correlations => we can calculated the
            # statistics directly
            ci_mean = self.substrate_probabilities
            cij_corr = np.diag(ci_mean - ci_mean**2)
            
        return ci_mean, cij_corr
    
    
    def mixture_statistics_brute_force(self):
        """ calculates mixture statistics using a brute force algorithm """
        
        Z = 0
        hist1d = np.zeros(self.Ns)
        hist2d = np.zeros((self.Ns, self.Ns))
        
        # iterate over all mixtures
        for c, weight_c in self._iterate_mixtures():
            Z += weight_c        
            hist1d += c * weight_c
            hist2d += np.outer(c, c) * weight_c
        
        # calculate the frequency and the correlations 
        ci_mean = hist1d / Z
        cij = hist2d / Z
        cij_corr = cij - np.outer(ci_mean, ci_mean)
        
        return ci_mean, cij_corr  
    
    
    def mixture_statistics_monte_carlo(self):
        """ calculates mixture statistics using a metropolis algorithm """
       
        count = 0
        hist1d = np.zeros(self.Ns, np.int)
        hist2d = np.zeros((self.Ns, self.Ns), np.int)

        # sample mixtures uniformly        
        for c in self._sample_mixtures():
            count += 1
            hist1d += c
            hist2d += np.outer(c, c)
        
        # calculate the frequency and the correlations 
        ci_mean = hist1d / count
        cij = hist2d / count
        cij_corr = cij - np.outer(ci_mean, ci_mean)
        
        return ci_mean, cij_corr
    
    
    def mixture_entropy(self):
        """ return the entropy in the mixture distribution """
        
        mixture_size = self.parameters['fixed_mixture_size']
                
        if self.has_correlations or mixture_size is not None:
            # complicated case => run brute force or monte carlo
            if self.Ns <= self.parameters['brute_force_threshold_Ns']:
                return self.mixture_entropy_brute_force()
            else:
                return self.mixture_entropy_monte_carlo()
        
        else:
            # simple case => calculate explicitly
            return -sum(ps*np.log2(ps) + (1 - ps)*np.log2(1 - ps)
                        for ps in self.substrate_probabilities)
        
    
    def mixture_entropy_brute_force(self):
        """ gets the entropy in the mixture distribution using brute force """
        Z, sum_wlogw = 0, 0
        
        for _, weight_c in self._iterate_mixtures():
            if weight_c > 0:
                Z += weight_c
                sum_wlogw += weight_c * np.log2(weight_c)
        return np.log2(Z) - sum_wlogw / Z
            

    def mixture_entropy_monte_carlo(self):
        """ gets the entropy in the mixture distribution using brute force """
        if self.Ns > 63:
            raise ValueError('Mixture entropy estimation only works for fewer '
                             'than 64 substrates.')
        
        # sample mixtures
        base = 2 ** np.arange(0, self.Ns)
        observations = collections.Counter()
        for c in self._sample_mixtures():
            observations[np.dot(c, base)] += 1
        
        # estimate entropy from the histogram
        counts = np.array(observations.values(), np.double)
        
        # naive implementation of measuring the entropy is
        #    ps = counts / self._sample_steps
        #    H = -np.sum(ps * np.log2(ps))
        # this can be transformed to a more stable implementation
            
        log_steps = np.log2(self._sample_steps)
        return -np.sum(counts*(np.log2(counts) - log_steps))/self._sample_steps
    

    def activity_single(self, method='auto'):
        """ calculates the average activity of each receptor
        
        `method` can be one of ['brute_force', 'monte_carlo', 'auto']. If 'auto'
            than the method is chosen automatically based on the problem size.
        """
        if method == 'auto':
            if self.Ns <= self.parameters['brute_force_threshold_Ns']:
                method = 'brute_force'
            else:
                method = 'monte_carlo'
                
        if method == 'brute_force' or method == 'brute-force':
            return self.activity_single_brute_force()
        elif method == 'monte_carlo' or method == 'monte-carlo':
            return self.activity_single_monte_carlo()
        elif method == 'estimate':
            return self.activity_single_estimate()
        else:
            raise ValueError('Unknown method `%s`.' % method)
        
        
    def activity_single_brute_force(self):
        """ calculates the average activity of each receptor """
        prob_a = np.zeros(self.Nr)
        Z = 0
        
        # iterate over all mixtures
        for c, prob_c in self._iterate_mixtures():
            # get the activity vector associated with m
            a = np.dot(self.int_mat, c).astype(np.bool)
            prob_a[a] += prob_c
            Z += prob_c
                
        # return the normalized output
        prob_a /= Z
        return prob_a

            
    def activity_single_monte_carlo(self):
        """ calculates the average activity of each receptor """ 
        count_a = np.zeros(self.Nr)
        for c in self._sample_mixtures():
            # choose a mixture vector according to substrate probabilities
            # get the associated output
            a = np.dot(self.int_mat, c).astype(np.bool)
            
            count_a[a] += 1
            
        # return the normalized output
        count_a /= self._sample_steps
        return count_a
            
    
    def activity_single_estimate(self, approx_prob=False):
        """ estimates the average activity of each receptor. 
        `approx_prob` determines whether the probabilities of encountering
            substrates in mixtures are calculated exactly or only approximative,
            which should work for small probabilities. """
        if self.has_correlations:
            raise NotImplementedError('Not implemented for correlated mixtures')

        S_ni = self.int_mat
        prob_s = self.substrate_probabilities

        # calculate the probabilities of exciting receptors and pairs
        if approx_prob:
            # approximate calculation for small prob_s
            q_n = np.dot(S_ni, prob_s)
            assert np.all(q_n <= 1)
            
        else:
            # proper calculation of the cluster probabilities
            q_n = np.zeros(self.Nr)
            S_ni_mask = S_ni.astype(np.bool)
            for a in range(self.Nr):
                q_n[a] = 1 - np.product(1 - prob_s[S_ni_mask[a, :]])
                
        return q_n


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
    
    
    def crosstalk(self, method='auto'):
        """ calculates the crosstalk between receptors
        
        `method` can be one of ['brute_force', 'monte_carlo', 'auto']. If 'auto'
            than the method is chosen automatically based on the problem size.
        """
        if method == 'auto':
            if self.Ns <= self.parameters['brute_force_threshold_Ns']:
                method = 'brute_force'
            else:
                method = 'monte_carlo'
                
        if method == 'brute_force' or method == 'brute-force':
            return self.crosstalk_brute_force()
        elif method == 'monte_carlo' or method == 'monte-carlo':
            return self.crosstalk_monte_carlo()
        elif method == 'estimate':
            return self.crosstalk_estimate()
        else:
            raise ValueError('Unknown method `%s`.' % method)
            
    
    def crosstalk_brute_force(self):
        """ calculates the crosstalk between receptors using brute force """
        q_nm = np.zeros((self.Nr, self.Nr))
        Z = 0
        
        # iterate over all mixtures
        for c, prob_c in self._iterate_mixtures():
            # get the activity vector associated with m
            a = np.dot(self.int_mat, c).astype(np.bool)
            q_nm += prob_c * np.outer(a, a)
            Z += prob_c
               
        # return the normalized output
        q_nm /= Z 
        return q_nm
    
            
    def crosstalk_monte_carlo(self):
        """ calculates the crosstalk between receptors using brute force """
        q_nm = np.zeros((self.Nr, self.Nr))
        for c in self._sample_mixtures():
            # get the output vector
            a_n = (np.dot(self.int_mat, c) >= 1)
            q_nm += np.outer(a_n, a_n)
        
        # return the normalized output
        q_nm /= self._sample_steps
        return q_nm 
    
            
    def crosstalk_estimate(self, approx_prob=False):
        """ calculates the estimated crosstalk between receptors """
        if self.has_correlations:
            raise NotImplementedError('Not implemented for correlated mixtures')

        S_ni = self.int_mat
        prob_s = self.substrate_probabilities
        
        if approx_prob:
            # approximate calculation for small prob_s
            q_nm = np.einsum('ai,bi,i->ab', S_ni, S_ni, prob_s)
            assert np.all(q_nm <= 1)
            
        else:
            # proper calculation of the probabilities
            q_nm = np.zeros((self.Nr, self.Nr))
            S_ni_mask = S_ni.astype(np.bool)
            for n in range(self.Nr):
                for m in range(self.Nr):
                    mask = S_ni_mask[n, :] * S_ni_mask[m, :]
                    q_nm[n, m] = 1 - np.product(1 - prob_s[mask])
            
        return q_nm


    def mutual_information(self, method='auto', **kwargs):
        """ calculate the mutual information.
        
        `method` can be one of ['brute_force', 'monte_carlo', 'auto']. If 'auto'
            than the method is chosen automatically based on the problem size.
        `ret_prob_activity` determines whether the probabilities of the
            different outputs are returned or not
        """
        if method == 'auto':
            if self.Ns <= self.parameters['brute_force_threshold_Ns']:
                method = 'brute_force'
            else:
                method = 'monte_carlo'
                
        if method == 'brute_force' or method == 'brute-force':
            return self.mutual_information_brute_force(**kwargs)
        elif method == 'monte_carlo' or method == 'monte-carlo':
            return self.mutual_information_monte_carlo(**kwargs)
        elif method == 'estimate':
            return self.mutual_information_estimate(**kwargs)
        else:
            raise ValueError('Unknown method `%s`.' % method)
            
            
    def mutual_information_brute_force(self, ret_prob_activity=False):
        """ calculate the mutual information by constructing all possible
        mixtures """
        base = 2 ** np.arange(0, self.Nr)

        # prob_a contains the probability of finding activity a as an output.
        prob_a = np.zeros(2**self.Nr)
        for c, prob_c in self._iterate_mixtures():
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
            return MI, prob_a
        else:
            return MI
            
            
    def mutual_information_monte_carlo(self, ret_error=False,
                                       ret_prob_activity=False,
                                       bias_correction=False):
        """ calculate the mutual information using a Monte Carlo strategy. """
        base = 2 ** np.arange(0, self.Nr)

        steps = self._sample_steps

        # sample mixtures according to the probabilities of finding
        # substrates
        count_a = np.zeros(2**self.Nr)
        for c in self._sample_mixtures():
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
        
        if bias_correction:
            # add entropy bias correction
            MI += (np.count_nonzero(prob_a) - 1)/(2*steps)
        
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
        """ calculate the mutual information using a Monte Carlo strategy. """
        if self.has_correlations:
            raise NotImplementedError('Not implemented for correlated mixtures')
                
        base = 2 ** np.arange(0, self.Nr)
        prob_s = self.substrate_probabilities

        max_steps = self._sample_steps
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

                    
    def mutual_information_estimate(self, approx_prob=False):
        """ returns a simple estimate of the mutual information.
        `approx_prob` determines whether the probabilities of encountering
            substrates in mixtures are calculated exactly or only approximative,
            which should work for small probabilities. """
            
        q_n = self.activity_single_estimate(approx_prob=approx_prob)
        q_nm = self.crosstalk_estimate(approx_prob=approx_prob)
                    
        # set diagonal to zero to simplify subsequent sums
        np.fill_diagonal(q_nm, 0)
                    
        # calculate the approximate mutual information
        MI = self.Nr
        for n in range(self.Nr):
            MI -= 0.5/LN2 * (1 - 2*q_n[n])**2
            for m in range(self.Nr):
                MI -= 1/LN2 * (0.75*q_nm[n, m] + q_n[n] + q_n[m] - 1)*q_nm[n, m]
                for l in range(self.Nr):
                    MI -= 0.5/LN2 * q_nm[n, m]*q_nm[m, l]
                
        return MI              
        
        
    def inefficiency_estimate(self):
        """ returns the estimated performance of the system, which acts as a
        proxy for the mutual information between input and output """
        if self.has_correlations:
            raise NotImplementedError('Not implemented for correlated mixtures')

        S_ni = self.int_mat
        prob_s = self.substrate_probabilities
        
        # collect the terms describing the activity entropy
        term_entropy = np.sum((0.5 - np.prod(1 - S_ni*prob_s, axis=1)) ** 2)
        
        # collect the terms describing the crosstalk
        mat_ab = np.einsum('ij,kj,j->ik', S_ni, S_ni, prob_s)
        term_crosstalk = 2*np.sum(mat_ab[np.triu_indices(self.Nr, 1)]) 
        
        # add up the terms to produce the inefficiency parameter
        crosstalk_weight = self.parameters['inefficiency_weight']
        return term_entropy + crosstalk_weight*term_crosstalk
        
        
    def optimize_library(self, target, method='descent', direction='max',
                         **kwargs):
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
            `descent_multiple`: gradient descent starting from multiple initial
                conditions.
            `anneal`: simulated annealing
        """
        if method == 'descent':
            return self.optimize_library_descent(target, direction, **kwargs)
        elif method == 'descent_multiple' or method == 'descent-multiple':
            return self.optimize_library_descent_multiple(target, direction,
                                                          **kwargs)
        elif method == 'anneal':
            return self.optimize_library_anneal(target, direction, **kwargs)
            
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
            # store extra information
            start_time = time.time()
            info = {'values': {}}
            values_count = self.parameters['optimizer_values_count']
            values_step = max(1, steps // values_count)
        
        if multiprocessing:
            # run the calculations in multiple processes
            pool = mp.Pool()
            pool_size = len(pool._pool)
            if ret_info:
                values_step = max(1, values_step // pool_size)
            
            # iterate for given number of steps
            for step in range(int(steps) // pool_size):
                joblist = []
                init_arguments = self.init_arguments
                for _ in range(pool_size):
                    # modify the current state and add it to the job list
                    i = random.randrange(self.int_mat.size)
                    self.int_mat.flat[i] = 1 - self.int_mat.flat[i]
                    init_arguments['parameters'].update({
                            'initialize_state': 'exact',
                            'interaction_matrix': self.int_mat
                        })
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
                        
                if ret_info and step % values_step == 0:
                    info['values'][step * pool_size] = results[res_best]
                
        else:
            # run the calculations in this process
            for step in range(int(steps)):
                # modify the current state
                i = random.randrange(self.int_mat.size)
                self.int_mat.flat[i] = 1 - self.int_mat.flat[i]
    
                # get the value of the new state
                value = target_function()
                
                improved = ((direction == 'max' and value > value_best) or
                            (direction == 'min' and value < value_best))
                if improved:
                    # save the state as the new best value
                    value_best, state_best = value, self.int_mat.copy()
                else:
                    # undo last change
                    self.int_mat.flat[i] = 1 - self.int_mat.flat[i]
                    
                if ret_info and step % values_step == 0:
                    info['values'][step] = value_best

        # sort the best state and store it in the current object
        state_best = self.sort_interaction_matrix(state_best)
        self.int_mat = state_best.copy()

        if ret_info:
            info['total_time'] = time.time() - start_time    
            info['states_considered'] = steps
            info['performance'] = steps / info['total_time']
            return value_best, state_best, info
        else:
            return value_best, state_best
 
             
    def optimize_library_descent_multiple(self, target, direction='max',
                                          trials=4, multiprocessing=False,
                                          **kwargs):
        """ optimizes the current library to maximize the result of the target
        function using gradient descent from `trials` different staring
        positions. Only the result from the best run will be returned """
        
        # pass some parameters down to the optimization function to call
        kwargs['target'] = target
        kwargs['direction'] = direction
        
        # initialize the list of jobs with an optimization job starting from the
        # current interaction matrix
        joblist = [(self.init_arguments, 'optimize_library_descent', kwargs)]
        int_mat = self.int_mat #< store matrix to restore it later

        # add additional jobs with random initial interaction matrices
        init_arguments = self.init_arguments
        for _ in range(trials - 1):
            # modify the current state and add it to the job list
            self.choose_interaction_matrix(density='auto')
            
            init_arguments['parameters'].update({
                    'initialize_state': 'exact',
                    'interaction_matrix': self.int_mat
                })
                                                            
            joblist.append((copy.deepcopy(init_arguments),
                            'optimize_library_descent', kwargs))
            
        # restore interaction matrix of this object
        self.int_mat = int_mat
        
        if multiprocessing:
            # calculate all results in parallel
            pool = mp.Pool()
            result_iter = pool.imap_unordered(_optimize_library_job, joblist)
        
        else:
            # create a generator over which we iterate later
            result_iter = (_optimize_library_job(job) for job in joblist)
        
        # find the best result by iterating over all results
        result_best = None
        for result in result_iter:
            # check whether this run improved the result
            if result_best is None:
                result_best = result
            elif ((direction == 'max' and result[0] > result_best[0]) or
                  (direction == 'min' and result[0] < result_best[0])):
                result_best = result
                
        # sort the best state and store it in the current object
        state = self.sort_interaction_matrix(result_best[1])
        self.int_mat = state.copy()

        return result_best
                               
    
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
        annealer = ReceptorOptimizerAnnealer(self, target, direction, args,
                                             ret_info=ret_info)
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
    # ... get the method to evaluate ...
    method = getattr(obj, args[1])
    # ... and evaluate it  
    if len(args) > 2:
        return method(**args[2])
    else:
        return method()

   
   
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
    
