'''
Created on Jan 16, 2015

@author: David Zwicker <dzwicker@seas.harvard.edu>

Module that monkey patches classes in other modules with equivalent, but faster
methods.
'''

from __future__ import division

import numba
import numpy as np

# these methods are used in getattr calls
from . import library_numeric
from utils.numba_patcher import NumbaPatcher, check_return_value_approx


NUMBA_NOPYTHON = True #< globally decide whether we use the nopython mode
NUMBA_NOGIL = True

# initialize the numba patcher and add methods one by one
numba_patcher = NumbaPatcher(module=library_numeric)



@numba.jit(nopython=NUMBA_NOPYTHON, nogil=NUMBA_NOGIL)
def LibraryBinaryNumeric_activity_single_brute_force_numba(
         Ns, Nr, int_mat, prob_s, ak, prob_a):
    """ calculates the average activity of each receptor """
    # iterate over all mixtures m
    for m in xrange(2**Ns):
        pm = 1     #< probability of finding this mixture
        ak[:] = 0  #< activity pattern of this mixture
        
        # iterate through substrates in the mixture
        for i in xrange(Ns):
            r = m % 2
            m //= 2
            if r == 1:
                # substrate i is present
                pm *= prob_s[i]
                for a in xrange(Nr):
                    if int_mat[a, i] == 1:
                        ak[a] = 1
            else:
                # substrate i is not present
                pm *= 1 - prob_s[i]
                
        # add probability to the active receptors
        for a in xrange(Nr):
            if ak[a] == 1:
                prob_a[a] += pm


def LibraryBinaryNumeric_activity_single_brute_force(self):
    """ calculates the average activity of each receptor """
    prob_a = np.zeros(self.Nr) 
    
    # call the jitted function
    LibraryBinaryNumeric_activity_single_brute_force_numba(
        self.Ns, self.Nr, self.int_mat,
        self.substrate_probabilities, #< prob_s
        np.empty(self.Nr, np.uint), #< ak
        prob_a
    )
    return prob_a


numba_patcher.register_method(
    'LibraryBinaryNumeric.activity_single_brute_force',
    LibraryBinaryNumeric_activity_single_brute_force,
)
    


@numba.jit(nopython=NUMBA_NOPYTHON, nogil=NUMBA_NOGIL)
def LibraryBinaryNumeric_activity_correlations_brute_force_numba(
        Ns, Nr, int_mat, prob_s, ak, prob_a):
    """ calculates the correlations between receptor activities """
    # iterate over all mixtures m
    for m in xrange(2**Ns):
        pm = 1     #< probability of finding this mixture
        ak[:] = 0  #< activity pattern of this mixture
        
        # iterate through substrates in the mixture
        for i in xrange(Ns):
            r = m % 2
            m //= 2
            if r == 1:
                # substrate i is present
                pm *= prob_s[i]
                for a in xrange(Nr):
                    if int_mat[a, i] == 1:
                        ak[a] = 1
            else:
                # substrate i is not present
                pm *= 1 - prob_s[i]
                
        # add probability to the active receptors
        for a in xrange(Nr):
            if ak[a] == 1:
                prob_a[a, a] += pm
                for b in xrange(a + 1, Nr):
                    if ak[b] == 1:
                        prob_a[a, b] += pm
                        prob_a[b, a] += pm
                    
    
def LibraryBinaryNumeric_activity_correlations_brute_force(self):
    """ calculates the correlations between receptor activities """
    prob_a = np.zeros((self.Nr, self.Nr)) 
    
    # call the jitted function
    LibraryBinaryNumeric_activity_correlations_brute_force_numba(
        self.Ns, self.Nr, self.int_mat,
        self.substrate_probabilities, #< prob_s
        np.empty(self.Nr, np.uint), #< ak
        prob_a
    )
    return prob_a


numba_patcher.register_method(
    'LibraryBinaryNumeric.activity_correlations_brute_force',
    LibraryBinaryNumeric_activity_correlations_brute_force
)
    
    

@numba.jit(nopython=NUMBA_NOPYTHON, nogil=NUMBA_NOGIL)
def LibraryBinaryNumeric_mutual_information_brute_force_numba(
        Ns, Nr, int_mat, prob_s, ak, prob_a):
    """ calculate the mutual information by constructing all possible
    mixtures """
    # iterate over all mixtures m
    for m in xrange(2**Ns):
        pm = 1     #< probability of finding this mixture
        ak[:] = 0  #< activity pattern of this mixture
        # iterate through substrates in the mixture
        for i in xrange(Ns):
            r = m % 2
            m //= 2
            if r == 1:
                # substrate i is present
                pm *= prob_s[i]
                for a in xrange(Nr):
                    if int_mat[a, i] == 1:
                        ak[a] = 1
            else:
                # substrate i is not present
                pm *= 1 - prob_s[i]
                
        # calculate the activity pattern id
        a_id, base = 0, 1
        for a in xrange(Nr):
            if ak[a] == 1:
                a_id += base
            base *= 2
        
        prob_a[a_id] += pm
    
    # calculate the mutual information from the observed probabilities
    MI = 0
    for pa in prob_a:
        if pa > 0:
            MI -= pa*np.log2(pa)
    
    return MI
    

def LibraryBinaryNumeric_mutual_information_brute_force(self, ret_prob_activity=False):
    """ calculate the mutual information by constructing all possible
    mixtures """
    prob_a = np.zeros(2**self.Nr) 
    
    # call the jitted function
    MI = LibraryBinaryNumeric_mutual_information_brute_force_numba(
        self.Ns, self.Nr, self.int_mat,
        self.substrate_probabilities, #< prob_s
        np.empty(self.Nr, np.uint), #< ak
        prob_a
    )
    
    if ret_prob_activity:
        return MI, prob_a
    else:
        return MI


numba_patcher.register_method(
    'LibraryBinaryNumeric.mutual_information_brute_force',
    LibraryBinaryNumeric_mutual_information_brute_force
)



@numba.jit(nopython=NUMBA_NOPYTHON, nogil=NUMBA_NOGIL) 
def LibraryBinaryNumeric_mutual_information_monte_carlo_numba(
        Ns, Nr, steps, int_mat, prob_s, ak, prob_a):
    """ calculate the mutual information using a monte carlo strategy. The
    number of steps is given by the model parameter 'monte_carlo_steps' """
        
    # sample mixtures according to the probabilities of finding
    # substrates
    for _ in xrange(steps):
        # choose a mixture vector according to substrate probabilities
        ak[:] = 0  #< activity pattern of this mixture
        for i in xrange(Ns):
            if np.random.random() < prob_s[i]:
                # the substrate i is present in the mixture
                for a in xrange(Nr):
                    if int_mat[a, i] == 1:
                        # receptor a is activated by substrate i
                        ak[a] = 1
        
        # calculate the activity pattern id
        a_id, base = 0, 1
        for a in xrange(Nr):
            if ak[a] == 1:
                a_id += base
            base *= 2
        
        # increment counter for this output
        prob_a[a_id] += 1
        
    # normalize the probabilities by the number of steps we did
    for k in xrange(len(prob_a)):
        prob_a[k] /= steps
    
    # calculate the mutual information from the observed probabilities
    MI = 0
    for pa in prob_a:
        if pa > 0:
            MI -= pa*np.log2(pa)
    
    return MI
    

def LibraryBinaryNumeric_mutual_information_monte_carlo(self, ret_error=False,
                                                        ret_prob_activity=False):
    """ calculate the mutual information by constructing all possible
    mixtures """
    prob_a = np.zeros(2**self.Nr) 
 
    # call the jitted function
    MI = LibraryBinaryNumeric_mutual_information_monte_carlo_numba(
        self.Ns, self.Nr, int(self.parameters['monte_carlo_steps']), 
        self.int_mat,
        self.substrate_probabilities, #< prob_s
        np.empty(self.Nr, np.uint), #< ak
        prob_a
    )
    
    if ret_error:
        # estimate the error of the mutual information calculation
        steps = int(self.parameters['monte_carlo_steps'])
        MI_err = sum(np.abs(1/np.log(2) + np.log2(pa)) * pa
                     for pa in prob_a if pa != 0) / np.sqrt(steps)

        if ret_prob_activity:
            return MI, MI_err, prob_a
        else:
            return MI, MI_err

    else:    
        # do not estimate the error of the mutual information calculation
        if ret_prob_activity:
            return MI, prob_a
        else:
            return MI


numba_patcher.register_method(
    'LibraryBinaryNumeric.mutual_information_monte_carlo',
    LibraryBinaryNumeric_mutual_information_monte_carlo,
    check_return_value_approx
)



@numba.jit(locals={'i_count': numba.int32}, nopython=NUMBA_NOPYTHON,
           nogil=NUMBA_NOGIL)
def LibraryBinaryNumeric_mutual_information_estimate_approx_numba(
        Ns, Nr, int_mat, prob_s, p_Ga, ids):
    """ calculate the mutual information by constructing all possible
    mixtures """
    
    MI = Nr
    # iterate over all receptors
    for a in xrange(Nr):
        # evaluate the direct
        i_count = 0 #< number of substrates that excite receptor a
        prob = 0
        for i in xrange(Ns):
            if int_mat[a, i] == 1:
                prob += prob_s[i]
                ids[i_count] = i
                i_count += 1
        p_Ga[a] = prob
        MI -= 0.5*(1 - 2*p_Ga[a])**2

        # iterate over all other receptors to estimate crosstalk
        for b in xrange(a):
            prod = 1
            for k in xrange(i_count):
                if int_mat[b, ids[k]] == 1:
                    prod *= 1 - prob_s[ids[k]]
            p_Gab = 1 - prod        

            MI -= 2*(1 - p_Ga[a] - p_Ga[b] + 3/4*p_Gab) * p_Gab
                
    return MI
    
    
@numba.jit(locals={'i_count': numba.int32}, nopython=NUMBA_NOPYTHON,
           nogil=NUMBA_NOGIL)
def LibraryBinaryNumeric_mutual_information_estimate_numba(
        Ns, Nr, int_mat, prob_s, p_Ga, ids):
    """ calculate the mutual information by constructing all possible
    mixtures """
    
    MI = Nr
    # iterate over all receptors
    for a in xrange(Nr):
        # evaluate the direct
        i_count = 0 #< number of substrates that excite receptor a
        prod = 1    #< product important for calculating the probabilities
        for i in xrange(Ns):
            if int_mat[a, i] == 1:
                prod *= 1 - prob_s[i]
                ids[i_count] = i
                i_count += 1
        p_Ga[a] = 1 - prod
        MI -= 0.5*(1 - 2*p_Ga[a])**2

        # iterate over all other receptors to estimate crosstalk
        for b in xrange(a):
            prod = 1
            for k in xrange(i_count):
                if int_mat[b, ids[k]] == 1:
                    prod *= 1 - prob_s[ids[k]]
            p_Gab = 1 - prod        

            MI -= 2*(1 - p_Ga[a] - p_Ga[b] + 3/4*p_Gab) * p_Gab
                
    return MI
    

def LibraryBinaryNumeric_mutual_information_estimate(self, approx_prob=False):
    """ calculate the mutual information by constructing all possible
    mixtures """
    if approx_prob:
        # call the jitted function that uses approximate probabilities
        MI = LibraryBinaryNumeric_mutual_information_estimate_approx_numba(
            self.Ns, self.Nr, self.int_mat,
            self.substrate_probabilities,  #< prob_s
            np.empty(self.Nr),           #< p_Ga
            np.empty(self.Ns, np.int32), #< ids
        )

    else:    
        # call the jitted function that uses exact probabilities
        MI = LibraryBinaryNumeric_mutual_information_estimate_numba(
            self.Ns, self.Nr, self.int_mat,
            self.substrate_probabilities,  #< prob_s
            np.empty(self.Nr),           #< p_Ga
            np.empty(self.Ns, np.int32), #< ids
        )
    
    return MI


numba_patcher.register_method(
    'LibraryBinaryNumeric.mutual_information_estimate',
    LibraryBinaryNumeric_mutual_information_estimate
)



@numba.jit(nopython=NUMBA_NOPYTHON, nogil=NUMBA_NOGIL)
def LibraryBinaryNumeric_inefficiency_estimate_numba(int_mat, prob_s,
                                                       crosstalk_weight):
    """ returns the estimated performance of the system, which acts as a
    proxy for the mutual information between input and output """
    Nr, Ns = int_mat.shape
    
    res = 0
    for a in xrange(Nr):
        activity_a = 1
        term_crosstalk = 0
        for i in xrange(Ns):
            if int_mat[a, i] == 1:
                # consider the terms describing the activity entropy
                activity_a *= 1 - prob_s[i]
                
                # consider the terms describing the crosstalk
                sum_crosstalk = 0
                for b in xrange(a + 1, Nr):
                    sum_crosstalk += int_mat[b, i]
                term_crosstalk += sum_crosstalk * prob_s[i] 

        res += (0.5 - activity_a)**2 + 2*crosstalk_weight * term_crosstalk
            
    return res


def LibraryBinaryNumeric_inefficiency_estimate(self):
    """ returns the estimated performance of the system, which acts as a
    proxy for the mutual information between input and output """
    prob_s = self.substrate_probabilities
    crosstalk_weight = self.parameters['inefficiency_weight']
    return LibraryBinaryNumeric_inefficiency_estimate_numba(self.int_mat, prob_s,
                                                              crosstalk_weight)


numba_patcher.register_method(
    'LibraryBinaryNumeric.inefficiency_estimate',
    LibraryBinaryNumeric_inefficiency_estimate
)

