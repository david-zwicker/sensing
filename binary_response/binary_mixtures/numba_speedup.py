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
def _mixture_energy(ci, hi, Jij):
    """ helper function that calculates the "energy" associated with the
    mixture `ci`, given commonness vector `hi` and correlation matrix `Jij` """ 
    energy = 0
    Ns = ci.size
    for i in range(Ns):
        if ci[i] > 0:
            energy += hi[i] + Jij[i, i]
            for j in range(i + 1, Ns):
                energy += 2 * Jij[i, j] * ci[j]
    return -energy


    
@numba.jit(nopython=NUMBA_NOPYTHON, nogil=NUMBA_NOGIL)
def _mixture_energy_indices(indices, hi, Jij):
    """ helper function that calculates the "energy" associated with the
    mixture defined by the `indices` of substrates that are present, given
    commonness vector `hi` and correlation matrix `Jij` """ 
    energy = 0
    for k, i in enumerate(indices):
        energy += hi[i] + Jij[i, i]
        for j in indices[k + 1:]:
            energy += 2 * Jij[i, j]
    return -energy



@numba.jit(nopython=NUMBA_NOPYTHON, nogil=NUMBA_NOGIL)
def _activity_pattern(int_mat, ci):
    """ helper function that calculates the id of the activity pattern from a
    given concentration vector and the associated interaction matrix """
    Nr, Ns = int_mat.shape
    
    # calculate the activity pattern id
    a_id, base = 0, 1
    for n in range(Nr):
        for i in range(Ns):
            if ci[i] * int_mat[n, i] == 1:
                # substrate is present and excites receptor
                a_id += base
                break
        base *= 2
    return a_id


    
#===============================================================================
# ACTIVITY SINGLE
#===============================================================================


@numba.jit(nopython=NUMBA_NOPYTHON, nogil=NUMBA_NOGIL)
def LibraryBinaryNumeric_activity_single_brute_force_corr_numba(
        int_mat, hi, Jij, ci, ak, prob_a):
    """ calculates the average activity of each receptor """
    Nr, Ns = int_mat.shape
    
    # iterate over all mixtures c
    Z = 0
    for c in range(2**Ns):
        # extract the mixture and the activity from the single integer `c`
        ak[:] = 0
        for i in range(Ns):
            ci[i] = c % 2
            c //= 2
        
            if ci[i] == 1:
                # determine which receptors this substrate activates
                for a in range(Nr):
                    if int_mat[a, i] == 1:
                        ak[a] = 1
        
        # calculate the probability of finding this mixture 
        pm = np.exp(-_mixture_energy(ci, hi, Jij))
        Z += pm
        
        # add probability to the active receptors
        for a in range(Nr):
            if ak[a] == 1:
                prob_a[a] += pm
        
    # normalize by partition sum        
    for a in range(Nr):
        prob_a[a] /= Z



@numba.jit(nopython=NUMBA_NOPYTHON, nogil=NUMBA_NOGIL)
def LibraryBinaryNumeric_activity_single_brute_force_numba(
         int_mat, prob_s, ak, prob_a):
    """ calculates the average activity of each receptor """
    Nr, Ns = int_mat.shape
    
    # iterate over all mixtures m
    for m in range(2**Ns):
        pm = 1     #< probability of finding this mixture
        ak[:] = 0  #< activity pattern of this mixture
        
        # iterate through substrates in the mixture
        for i in range(Ns):
            r = m % 2
            m //= 2
            if r == 1:
                # substrate i is present
                pm *= prob_s[i]
                for a in range(Nr):
                    if int_mat[a, i] == 1:
                        ak[a] = 1
            else:
                # substrate i is not present
                pm *= 1 - prob_s[i]
                
        # add probability to the active receptors
        for a in range(Nr):
            if ak[a] == 1:
                prob_a[a] += pm



def LibraryBinaryNumeric_activity_single_brute_force(self):
    """ calculates the average activity of each receptor """
    prob_a = np.zeros(self.Nr) 
    
    if self.parameters['fixed_mixture_size'] is not None:
        raise NotImplementedError
    
    if self.has_correlations:
        # call the jitted function for correlated mixtures
        LibraryBinaryNumeric_activity_single_brute_force_corr_numba(
            self.int_mat,
            self.commonness, self.correlations, #< hi, Jij
            np.empty(self.Ns, np.uint), #< ci
            np.empty(self.Nr, np.uint), #< ak
            prob_a
        )
        
    else:
        # call the jitted function for uncorrelated mixtures
        LibraryBinaryNumeric_activity_single_brute_force_numba(
            self.int_mat,
            self.substrate_probabilities, #< prob_s
            np.empty(self.Nr, np.uint), #< ak
            prob_a
        )
        
    return prob_a



numba_patcher.register_method(
    'LibraryBinaryNumeric.activity_single_brute_force',
    LibraryBinaryNumeric_activity_single_brute_force,
)
    
    
#===============================================================================
# ACTIVITY CORRELATIONS
#===============================================================================


@numba.jit(nopython=NUMBA_NOPYTHON, nogil=NUMBA_NOGIL)
def LibraryBinaryNumeric_activity_correlations_brute_force_numba(
        Ns, Nr, int_mat, prob_s, ak, prob_a):
    """ calculates the correlations between receptor activities """
    # iterate over all mixtures m
    for m in range(2**Ns):
        pm = 1     #< probability of finding this mixture
        ak[:] = 0  #< activity pattern of this mixture
        
        # iterate through substrates in the mixture
        for i in range(Ns):
            r = m % 2
            m //= 2
            if r == 1:
                # substrate i is present
                pm *= prob_s[i]
                for a in range(Nr):
                    if int_mat[a, i] == 1:
                        ak[a] = 1
            else:
                # substrate i is not present
                pm *= 1 - prob_s[i]
                
        # add probability to the active receptors
        for a in range(Nr):
            if ak[a] == 1:
                prob_a[a, a] += pm
                for b in range(a + 1, Nr):
                    if ak[b] == 1:
                        prob_a[a, b] += pm
                        prob_a[b, a] += pm
                    
    
    
def LibraryBinaryNumeric_activity_correlations_brute_force(self):
    """ calculates the correlations between receptor activities """
    if self.has_correlations:
        raise NotImplementedError('Not implemented for correlated mixtures')

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

    
#===============================================================================
# MUTUAL INFORMATION BRUTE FORCE
#===============================================================================

            
@numba.jit(locals={'i': numba.int32, 'j': numba.int32},
           nopython=NUMBA_NOPYTHON, nogil=NUMBA_NOGIL)
def LibraryBinaryNumeric_mutual_information_brute_force_fixed_numba(
        Ns, Nr, int_mat, hi, Jij, m, indices, prob_a):
    """ calculate the mutual information by constructing all possible
    mixtures """
    # initialize the mixture vector
    for i in range(m):
        indices[i] = i

    # iterate over all mixtures with a given number of substrates    
    Z = 0
    running = True
    while running:
        # find the next iteration of the mixture
        for i in range(m - 1, -1, -1):
            if indices[i] + m != i + Ns:
                indices[i] += 1
                for j in range(i + 1, m):
                    indices[j] = indices[j - 1] + 1
                break
        else:
            # set the last mixture
            for i in range(m):
                indices[i] = i
            running = False
        # `indices` now holds the indices of ones in the concentration vector

        # calculate the probability of finding this mixture 
        pm = np.exp(-_mixture_energy_indices(indices, hi, Jij))
        Z += pm

        # determine the activity pattern 
        a_id, base = 0, 1
        for a in range(Nr): 
            for i in indices:
                if int_mat[a, i] == 1:
                    a_id += base
                    break
            base *= 2
        
        prob_a[a_id] += pm
    
    # calculate the mutual information from the observed probabilities
    MI = 0
    for pa in prob_a:
        if pa > 0:
            pa /= Z #< normalize the probability distribution
            MI -= pa * np.log2(pa)
    
    return MI
        
   

@numba.jit(nopython=NUMBA_NOPYTHON, nogil=NUMBA_NOGIL)
def LibraryBinaryNumeric_mutual_information_brute_force_corr_numba(
        Ns, Nr, int_mat, hi, Jij, ci, ak, prob_a):
    """ calculate the mutual information by constructing all possible
    mixtures """
    Z = 0
    
    # iterate over all mixtures m
    for c in range(2**Ns):
        # extract the mixture and the activity from the single integer `c`
        ak[:] = 0
        for i in range(Ns):
            ci[i] = c % 2
            c //= 2
        
            if ci[i] == 1:
                # determine which receptors this substrate activates
                for a in range(Nr):
                    if int_mat[a, i] == 1:
                        ak[a] = 1
            
        # calculate the probability of finding this mixture 
        pm = np.exp(-_mixture_energy(ci, hi, Jij))
        Z += pm
        
        # calculate the activity pattern id
        a_id, base = 0, 1
        for a in range(Nr):
            if ak[a] == 1:
                a_id += base
            base *= 2
        
        prob_a[a_id] += pm
    
    # calculate the mutual information from the observed probabilities
    MI = 0
    for pa in prob_a:
        if pa > 0:
            pa /= Z #< normalize the probability distribution
            MI -= pa*np.log2(pa)
    
    return MI
        


@numba.jit(nopython=NUMBA_NOPYTHON, nogil=NUMBA_NOGIL)
def LibraryBinaryNumeric_mutual_information_brute_force_numba(
        Ns, Nr, int_mat, prob_s, ak, prob_a):
    """ calculate the mutual information by constructing all possible
    mixtures """
    # iterate over all mixtures m
    for m in range(2**Ns):
        pm = 1     #< probability of finding this mixture
        ak[:] = 0  #< activity pattern of this mixture
        # iterate through substrates in the mixture
        for i in range(Ns):
            r = m % 2
            m //= 2
            if r == 1:
                # substrate i is present
                pm *= prob_s[i]
                for a in range(Nr):
                    if int_mat[a, i] == 1:
                        ak[a] = 1
            else:
                # substrate i is not present
                pm *= 1 - prob_s[i]
                
        # calculate the activity pattern id
        a_id, base = 0, 1
        for a in range(Nr):
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
    mixture_size = self.parameters['fixed_mixture_size']
    
    if mixture_size is not None:
        # call the jitted function for mixtures with fixed size
        MI = LibraryBinaryNumeric_mutual_information_brute_force_fixed_numba(
            self.Ns, self.Nr, self.int_mat,
            self.commonness, self.correlations, #< hi, Jij
            int(mixture_size),
            np.empty(mixture_size, np.uint), #< inidices
            prob_a
        )
    
    elif self.has_correlations:
        # call the jitted function for correlated mixtures
        MI = LibraryBinaryNumeric_mutual_information_brute_force_corr_numba(
            self.Ns, self.Nr, self.int_mat,
            self.commonness, self.correlations, #< hi, Jij
            np.empty(self.Ns, np.uint), #< ci
            np.empty(self.Nr, np.uint), #< ak
            prob_a
        )
        
    else:
        # call the jitted function for uncorrelated mixtures
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

    
#===============================================================================
# MUTUAL INFORMATION MONTE CARLO
#===============================================================================


@numba.jit(nopython=NUMBA_NOPYTHON, nogil=NUMBA_NOGIL) 
def LibraryBinaryNumeric_mutual_information_monte_carlo_numba(
        Ns, Nr, steps, int_mat, prob_s, ak, prob_a):
    """ calculate the mutual information using a monte carlo strategy. The
    number of steps is given by the model parameter 'monte_carlo_steps' """
        
    # sample mixtures according to the probabilities of finding substrates
    for _ in range(steps):
        # choose a mixture vector according to substrate probabilities
        ak[:] = 0  #< activity pattern of this mixture
        for i in range(Ns):
            if np.random.random() < prob_s[i]:
                # the substrate i is present in the mixture
                for a in range(Nr):
                    if int_mat[a, i] == 1:
                        # receptor a is activated by substrate i
                        ak[a] = 1
        
        # calculate the activity pattern id
        a_id, base = 0, 1
        for a in range(Nr):
            if ak[a] == 1:
                a_id += base
            base *= 2
        
        # increment counter for this output
        prob_a[a_id] += 1
        
    # normalize the probabilities by the number of steps we did
    for k in range(len(prob_a)):
        prob_a[k] /= steps
    
    # calculate the mutual information from the observed probabilities
    MI = 0
    for pa in prob_a:
        if pa > 0:
            MI -= pa*np.log2(pa)
    
    return MI
 
 

@numba.jit(nopython=NUMBA_NOPYTHON, nogil=NUMBA_NOGIL) 
def LibraryBinaryNumeric_mutual_information_metropolis_numba(
        Ns, Nr, steps, int_mat, hi, Jij, ci, prob_a):
    """ calculate the mutual information using a monte carlo strategy. The
    number of steps is given by the model parameter 'monte_carlo_steps' """
        
    # initialize the concentration vector
    for i in range(Ns):
        ci[i] = np.random.randint(2) #< set to either 0 or 1
    E_last = _mixture_energy(ci, hi, Jij)
    a_id = _activity_pattern(int_mat, ci)
        
    # sample mixtures according to the probabilities of finding substrates
    for _ in range(steps):
        # choose a new mixture based on the old one
        k = np.random.randint(Ns)
        ci[k] = 1 - ci[k]
        E_new = _mixture_energy(ci, hi, Jij)
        
        if E_new < E_last or np.random.random() < np.exp(E_last - E_new):
            # accept the new state
            E_last = E_new

            # calculate the activity pattern from this mixture vector
            a_id = _activity_pattern(int_mat, ci)
        
        else:
            # reject the new state and revert to the last one
            ci[k] = 1 - ci[k]
        
        # increment counter for this output
        prob_a[a_id] += 1
        
    # normalize the probabilities by the number of steps we did
    for k in range(len(prob_a)):
        prob_a[k] /= steps
    
    # calculate the mutual information from the observed probabilities
    MI = 0
    for pa in prob_a:
        if pa > 0:
            MI -= pa*np.log2(pa)
    
    return MI
 
 

@numba.jit(nopython=NUMBA_NOPYTHON, nogil=NUMBA_NOGIL) 
def LibraryBinaryNumeric_mutual_information_metropolis_swap_numba(
        Ns, Nr, steps, int_mat, hi, Jij, mixture_size, ci, prob_a):
    """ calculate the mutual information using a monte carlo strategy. The
    number of steps is given by the model parameter 'monte_carlo_steps' """
    # find out how many zeros and ones there are => these numbers are fixed
    num_0 = Ns - mixture_size
    num_1 = mixture_size
    
    if num_0 == 0 or num_1 == 0:
        # there will be only a single mixture
        return 0

    # set the energy of the first substrate to infinity to make sure that the
    # first modified substrate is accepted        
    E_last = np.inf
        
    # sample mixtures according to the probabilities of finding
    # substrates
    for _ in range(steps):
        # choose two substrates to swap. Here, we choose the want_0-th zero and
        # the want_1-th one in the vector and swap these two
        want_0 = np.random.randint(num_0)
        want_1 = np.random.randint(num_1)
        count_0, count_1 = 0, 0 #< count how many zeros and ones we found
        i0, i1 = -1, -1 #< these will store the indices of the k-th 0 and 1
        for i in range(Ns):
            if ci[i] == 0:
                if i0 < 0:
                    # we are still looking for the index of the zero
                    if count_0 == want_0:
                        # found the substrate that we chose => switch it to a 1
                        i0 = i
                        ci[i0] = 1 #< toggle this entry 
                        if i1 >= 0: #< check if the one was already found
                            break
                    else:
                        count_0 += 1
    
            else: # ci[i] == 1
                if i1 < 0:
                    # we are still looking for the index of the one
                    if count_1 == want_1:
                        # found the substrate that we chose => switch it to a 0
                        i1 = i
                        ci[i1] = 0 #< toggle this entry
                        if i0 >= 0: #< check if the zero was already found
                            break
                    else:
                        count_1 += 1
        # we now switched the presence of substrate i0 and i1
        
        # calculate the energy of the new mixture
        E_new = _mixture_energy(ci, hi, Jij)
        
        if E_new < E_last or np.random.random() < np.exp(E_last - E_new):
            # accept the new mixture vector and save its energy
            E_last = E_new
                        
            # calculate the activity pattern id
            a_id = _activity_pattern(int_mat, ci)
            
        else:
            # reject the new state and revert to the last one  -> we can also
            # reuse the calculations of the activity pattern from the last step
            ci[i0] = 0
            ci[i1] = 1
        
        # increment counter for this output
        prob_a[a_id] += 1
        
    # normalize the probabilities by the number of steps we did
    for k in range(len(prob_a)):
        prob_a[k] /= steps
    
    # calculate the mutual information from the observed probabilities
    MI = 0
    for pa in prob_a:
        if pa > 0:
            MI -= pa*np.log2(pa)
    
    return MI

   

def LibraryBinaryNumeric_mutual_information_monte_carlo(self, ret_error=False,
                                                        ret_prob_activity=False,
                                                        bias_correction=False):
    """ calculate the mutual information by constructing all possible
    mixtures """
    prob_a = np.zeros(2**self.Nr)
    mixture_size = self.parameters['fixed_mixture_size']
 
    if mixture_size is not None:
        # use the version of the metropolis algorithm that keeps the number
        # of substrates in a mixture constant
        mixture_size = int(mixture_size)
        steps = self.get_steps('metropolis')
        
        # create random concentration vector with fixed substrate count
        ci = np.r_[np.ones(mixture_size, np.uint8),
                   np.zeros(self.Ns - mixture_size, np.uint8)]
        np.random.shuffle(ci)
        
        # call jitted function implementing swapping metropolis algorithm
        MI = LibraryBinaryNumeric_mutual_information_metropolis_swap_numba(
            self.Ns, self.Nr, steps, 
            self.int_mat,
            self.commonness, self.correlations, #< hi, Jij
            mixture_size, ci, prob_a
        )
    
    elif self.has_correlations:
        # mixture has correlations and we thus use a metropolis algorithm
        steps = self.get_steps('metropolis')
        
        # call jitted function implementing simple metropolis algorithm
        MI = LibraryBinaryNumeric_mutual_information_metropolis_numba(
            self.Ns, self.Nr, steps, 
            self.int_mat,
            self.commonness, self.correlations, #< hi, Jij
            np.empty(self.Ns, np.uint8), #< ci
            prob_a
        )
    
    else:
        # simple case without correlations and unconstrained number of ligands
        steps = self.get_steps('monte_carlo')
        
        # call jitted function implementing simple monte carlo algorithm
        MI = LibraryBinaryNumeric_mutual_information_monte_carlo_numba(
            self.Ns, self.Nr, steps, 
            self.int_mat,
            self.substrate_probabilities, #< prob_s
            np.empty(self.Nr, np.uint), #< ak
            prob_a
        )
        
    if bias_correction:
        # add entropy bias correction, MLE of [Paninski2003]
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

    
#===============================================================================
# MUTUAL INFORMATION ESTIMATION
#===============================================================================


@numba.jit(locals={'i_count': numba.int32}, nopython=NUMBA_NOPYTHON,
           nogil=NUMBA_NOGIL)
def LibraryBinaryNumeric_mutual_information_estimate_approx_numba(
        Ns, Nr, int_mat, prob_s, p_Ga, ids):
    """ calculate the mutual information by constructing all possible
    mixtures """
    
    MI = Nr
    # iterate over all receptors
    for a in range(Nr):
        # evaluate the direct
        i_count = 0 #< number of substrates that excite receptor a
        prob = 0
        for i in range(Ns):
            if int_mat[a, i] == 1:
                prob += prob_s[i]
                ids[i_count] = i
                i_count += 1
        p_Ga[a] = prob
        MI -= 0.5*(1 - 2*p_Ga[a])**2

        # iterate over all other receptors to estimate crosstalk
        for b in range(a):
            p_Gab = 0
            for k in range(i_count):
                if int_mat[b, ids[k]] == 1:
                    p_Gab += prob_s[ids[k]]

            MI -= 2*abs(1 - p_Ga[a] - p_Ga[b] + 3/4*p_Gab) * p_Gab
                
    return MI
    
    
    
@numba.jit(locals={'i_count': numba.int32}, nopython=NUMBA_NOPYTHON,
           nogil=NUMBA_NOGIL)
def LibraryBinaryNumeric_mutual_information_estimate_numba(
        Ns, Nr, int_mat, prob_s, p_Ga, ids):
    """ calculate the mutual information by constructing all possible
    mixtures """
    
    MI = Nr
    # iterate over all receptors
    for a in range(Nr):
        # evaluate the direct
        i_count = 0 #< number of substrates that excite receptor a
        prod = 1    #< product important for calculating the probabilities
        for i in range(Ns):
            if int_mat[a, i] == 1:
                prod *= 1 - prob_s[i]
                ids[i_count] = i
                i_count += 1
        p_Ga[a] = 1 - prod
        MI -= 0.5*(1 - 2*p_Ga[a])**2

        # iterate over all other receptors to estimate crosstalk
        for b in range(a):
            prod = 1
            for k in range(i_count):
                if int_mat[b, ids[k]] == 1:
                    prod *= 1 - prob_s[ids[k]]
            p_Gab = 1 - prod        

            MI -= 2*abs(1 - p_Ga[a] - p_Ga[b] + 3/4*p_Gab) * p_Gab
                
    return MI
    


def LibraryBinaryNumeric_mutual_information_estimate(self, approx_prob=False):
    """ calculate the mutual information by constructing all possible
    mixtures """
    if self.has_correlations:
        raise NotImplementedError('Not implemented for correlated mixtures')

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
    for a in range(Nr):
        activity_a = 1
        term_crosstalk = 0
        for i in range(Ns):
            if int_mat[a, i] == 1:
                # consider the terms describing the activity entropy
                activity_a *= 1 - prob_s[i]
                
                # consider the terms describing the crosstalk
                sum_crosstalk = 0
                for b in range(a + 1, Nr):
                    sum_crosstalk += int_mat[b, i]
                term_crosstalk += sum_crosstalk * prob_s[i] 

        res += (0.5 - activity_a)**2 + 2*crosstalk_weight * term_crosstalk
            
    return res



def LibraryBinaryNumeric_inefficiency_estimate(self):
    """ returns the estimated performance of the system, which acts as a
    proxy for the mutual information between input and output """
    if self.has_correlations:
        raise NotImplementedError('Not implemented for correlated mixtures')

    prob_s = self.substrate_probabilities
    crosstalk_weight = self.parameters['inefficiency_weight']
    return LibraryBinaryNumeric_inefficiency_estimate_numba(self.int_mat, prob_s,
                                                            crosstalk_weight)



numba_patcher.register_method(
    'LibraryBinaryNumeric.inefficiency_estimate',
    LibraryBinaryNumeric_inefficiency_estimate
)

