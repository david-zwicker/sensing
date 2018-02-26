'''
Created on Jan 16, 2015

@author: David Zwicker <dzwicker@seas.harvard.edu>

Module that monkey patches classes in other modules with equivalent, but faster
methods.
'''

from __future__ import division

import logging

import numba
from scipy import integrate, LowLevelCallable

# these methods are used in getattr calls
from . import pc_theory
from utils.numba.patcher import NumbaPatcher, check_return_value_approx
from utils.numba.tools import lognorm_cdf, lognorm_pdf


NUMBA_NOPYTHON = True #< globally decide whether we use the nopython mode
NUMBA_NOGIL = True

# initialize the numba patcher and add methods one by one
numba_patcher = NumbaPatcher(module=pc_theory)



@numba.cfunc("float64(int32, CPointer(float64))", nopython=True)
def nb_integrand_on(n_arg, x):
    """ compiled helper function for determining the probability that a channel
    turns on """
    # extract parameters
    assert n_arg == 6
    e_1 = x[0]
    mean_b, var_b = x[1], x[2]
    mean_t, var_t = x[3], x[4]
    gamma_2 = x[5]

    # calculate argument
    sf = 1 - lognorm_cdf(gamma_2 - e_1, mean_t, var_t)
    pdf = lognorm_pdf(e_1, mean_b, var_b)
    return sf * pdf


@numba.cfunc("float64(int32, CPointer(float64))", nopython=True)
def nb_integrand_off(n_arg, x):
    """ compiled helper function for determining the probability that a channel
    turns off """
    # extract parameters
    assert n_arg == 6
    e_1 = x[0]
    mean_b, var_b = x[1], x[2]
    mean_t, var_t = x[3], x[4]
    gamma_2 = x[5]

    # calculate argument
    cdf = lognorm_cdf(gamma_2 - e_1, mean_t, var_t)
    pdf = lognorm_pdf(e_1, mean_b, var_b)
    return cdf * pdf


def PrimacyCodingTheory_activity_distance_from_distributions_quad(self,
                         en_dist_background, en_dist_target, gamma_1, gamma_2):
    """ numerically solves the integrals for the probabilities of a channel
    becoming active and inactive. Returns the two probabilities. """
    if (en_dist_background.dist.name != 'lognorm' or
            en_dist_target.dist.name != 'lognorm'):
        logging.warning('Numba code only implemented for log-normal '
                        'distributions. Falling back to pure-python method.')
        this = PrimacyCodingTheory_activity_distance_from_distributions_quad
        return this._python_function(self, en_dist_background,
                                     en_dist_target, gamma_1, gamma_2)

    # extract statistics
    mean_b = en_dist_background.mean()
    var_b = en_dist_background.var()
    mean_t = en_dist_target.mean()
    var_t = en_dist_target.var()
    
    # determine the probability that a channel turns on and off, respectively
    p_on = integrate.quad(
            LowLevelCallable(nb_integrand_on.ctypes), 0, gamma_1,
            args=(mean_b, var_b, mean_t, var_t, gamma_2)
        )[0]
        
    if gamma_2 > gamma_1:
        p_off = integrate.quad(
                LowLevelCallable(nb_integrand_off.ctypes), gamma_1, gamma_2,
                args=(mean_b, var_b, mean_t, var_t, gamma_2)
            )[0]
    else:
        p_off = 0
    
    return p_on, p_off        


numba_patcher.register_method(
    'PrimacyCodingTheory._activity_distance_from_distributions_quad',
    PrimacyCodingTheory_activity_distance_from_distributions_quad,
    check_return_value_approx
)

