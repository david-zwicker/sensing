'''
Created on Apr 1, 2015

@author: David Zwicker <dzwicker@seas.harvard.edu>
'''

from __future__ import division

import functools
import logging
import multiprocessing as mp

import numpy as np
from scipy import special

from utils.misc import xlog2x


LN2 = np.log(2)

# define vectorize function for double results to use as a decorator
vectorize_double = functools.partial(np.vectorize, otypes=[np.double])



class LibraryBase(object):
    """ represents a single receptor library. This is a base class that provides
    general functionality and parameter management.
    
    For instance, the class provides a framework for calculating ensemble
    averages, where each time new commonness vectors are chosen randomly
    according to the parameters of the last call to `set_commonness`.  
    """

    # default parameters that are used to initialize a class if not overwritten
    parameters_default = {
        'initialize_state': 'auto',      #< how to initialize the state
        'ensemble_average_num': 32,      #< repetitions for ensemble average
        'multiprocessing_cores': 'auto', #< number of cores to use
    }


    def __init__(self, num_substrates, num_receptors, parameters=None):
        """ initialize a receptor library by setting the number of receptors,
        the number of substrates it can respond to, and optional additional
        parameters in the parameter dictionary """
        self.Ns = num_substrates
        self.Nr = num_receptors

        # initialize parameters with default ones from all parent classes
        self.parameters = {}
        for cls in reversed(self.__class__.__mro__):
            if hasattr(cls, 'parameters_default'):
                self.parameters.update(cls.parameters_default)
        # update parameters with the supplied ones
        if parameters is not None:
            self.parameters.update(parameters)


    @property
    def repr_params(self):
        """ return the important parameters that are shown in __repr__ """
        return ['Ns=%d' % self.Ns, 'Nr=%d' % self.Nr]


    def __repr__(self):
        params = ', '.join(self.repr_params)
        return '%s(%s)' % (self.__class__.__name__, params)


    @property
    def init_arguments(self):
        """ return the parameters of the model that can be used to reconstruct
        it by calling the __init__ method with these arguments """
        return {'num_substrates': self.Ns,
                'num_receptors': self.Nr,
                'parameters': self.parameters}

    
    def copy(self):
        """ returns a copy of the current object """
        return self.__class__(**self.init_arguments)


    @classmethod
    def from_other(cls, other, **kwargs):
        """ creates an instance of this class by using parameters from another
        instance """
        # create object with parameters from other object
        init_arguments = other.init_arguments
        init_arguments.update(kwargs)
        return cls(**init_arguments)


    @classmethod
    def get_random_arguments(cls, num_substrates=None, num_receptors=None):
        """ create random arguments for creating test instances """
        if num_substrates is None:
            num_substrates = np.random.randint(3, 6)
        if num_receptors is None:
            num_receptors =  np.random.randint(2, 4)
        
        # return the dictionary
        return {'num_substrates': num_substrates,
                'num_receptors': num_receptors}


    @classmethod
    def create_test_instance(cls, **kwargs):
        """ creates a test instance used for consistency tests """
        return cls(**cls.get_random_arguments(**kwargs))
  
            
    def get_number_of_cores(self):
        """ returns the number of cores to use in multiprocessing """
        multiprocessing_cores = self.parameters['multiprocessing_cores']
        if multiprocessing_cores == 'auto':
            return mp.cpu_count()
        else:
            return multiprocessing_cores
            
            
    def ensemble_average(self, method, avg_num=None, multiprocessing=False, 
                         ret_all=False, args=None):
        """ calculate an ensemble average of the result of the `method` of
        multiple different receptor libraries """
        
        if avg_num is None:
            avg_num = self.parameters['ensemble_average_num']
        if args is None:
            args = {}
        
        if multiprocessing and avg_num > 1:
            
            init_arguments = self.init_arguments
            
            # set the initialization procedure to ensemble, such that new
            # realizations are chosen at each iteration 
            if init_arguments['parameters']['initialize_state'] == 'auto':
                init_arguments['parameters']['initialize_state'] = 'ensemble'
            
            # run the calculations in multiple processes
            arguments = (self.__class__, init_arguments, method, args)
            pool = mp.Pool(processes=self.get_number_of_cores())
            result = pool.map(_ensemble_average_job, [arguments] * avg_num)
            
            # Apparently, multiprocessing sometimes opens too many files if
            # processes are launched to quickly and the garbage collector cannot
            # keep up. We thus explicitly terminate the pool here.
            pool.terminate()
            
        else:
            # run the calculations in this process
            cls = self.__class__
            result = [getattr(cls(**self.init_arguments), method)(**args)
                      for _ in range(avg_num)]
    
        # collect the results and calculate the statistics
        if ret_all:
            return result
        else:
            result = np.array(result)
            return result.mean(axis=0), result.std(axis=0)


    def ctot_statistics(self, **kwargs):
        """ returns the statistics for the total concentration. All arguments
        are passed to the call to `self.concentration_statistics` """
        # get the statistics of the individual substrates
        c_stats = self.concentration_statistics(**kwargs)
        
        # calculate the statistics of their sum
        ctot_mean = c_stats['mean'].sum()
        if c_stats.get('cov_is_diagonal', False):
            ctot_var = c_stats['var'].sum()
        else:
            ctot_var = c_stats['cov'].sum()
        
        return {'mean': ctot_mean, 'std': np.sqrt(ctot_var), 'var': ctot_var}
    
            
    def _estimate_qn_from_en(self, en_stats, excitation_model='default'):
        """ estimates probability q_n that a receptor is activated by a mixture
        based on the statistics of the excitations en """

        if excitation_model == 'default' or excitation_model is None:
            excitation_model = 'log-normal'

        if 'gauss' in excitation_model:
            if 'approx' in excitation_model:
                # estimate from a simple expression, which was obtained from
                # expanding the expression from the Gaussian
                q_n = _estimate_qn_from_en_gaussian_approx(en_stats['mean'],
                                                           en_stats['var'])
            else:
                # estimate from a gaussian distribution
                q_n = _estimate_qn_from_en_gaussian(en_stats['mean'],
                                                    en_stats['var'])

        elif 'log-normal' in excitation_model or 'lognorm' in excitation_model:
            if 'approx' in excitation_model:
                # estimate from a simple expression, which was obtained from
                # expanding the expression from the log-normal
                q_n = _estimate_qn_from_en_lognorm_approx(en_stats['mean'],
                                                          en_stats['var'])
            else:
                # estimate from a log-normal distribution
                q_n = _estimate_qn_from_en_lognorm(en_stats['mean'],
                                                   en_stats['var'])

        else:
            raise ValueError('Unknown excitation model `%s`' % excitation_model)
            
        return q_n
   
    
    def _estimate_qnm_from_en(self, en_stats):
        """ estimates crosstalk q_nm based on the statistics of the excitations
        en """
        en_cov = en_stats['cov']
        
        # calculate the correlation coefficient
        if np.isscalar(en_cov):
            # scalar case
            en_var = en_stats['var']
            if np.isclose(en_var, 0):
                rho = 0
            else:
                rho = en_cov / en_var
            
        else:
            # matrix case
            en_std = en_stats['std'] 
            with np.errstate(divide='ignore', invalid='ignore'):
                rho = np.divide(en_cov, np.outer(en_std, en_std))
    
            # replace values that are nan with zero. This might not be exact,
            # but only occurs in corner cases that are not interesting to us  
            rho[np.isnan(rho)] = 0
            
        # estimate the crosstalk
        q_nm = rho / (2*np.pi)
            
        return q_nm
    
            
    def _estimate_MI_from_q_values(self, q_n, q_nm, use_polynom=True):
        """ estimate the mutual information from given probabilities """
        # calculate the approximate mutual information from data
        if use_polynom:
            # use the quadratic approximation of the mutual information
            MI = self.Nr - 0.5/LN2 * np.sum((2*q_n - 1)**2)
            # calculate the crosstalk
            MI -= 8/LN2 * np.sum(np.triu(q_nm, 1)**2)
            
        else:
            # use the better approximation based on the formula published in
            #     V. Sessak and R. Monasson, J Phys A, 42, 055001 (2009) 
            MI = -np.sum(xlog2x(q_n) + xlog2x(1 - q_n))
        
            # calculate the crosstalk
            q_n2 = q_n**2 - q_n
            with np.errstate(divide='ignore', invalid='ignore'):
                q_nm_scaled = q_nm**2 / np.outer(q_n2, q_n2)
            
            # replace values that are not finite with zero. This might not be
            # exact, but only occurs in cases that are not interesting to us  
            q_nm_scaled[~np.isfinite(q_nm_scaled)] = 0
            
            MI -= 0.5/LN2 * np.sum(np.triu(q_nm_scaled, 1))
            
        return MI
    
        
    def _estimate_MI_from_q_stats(self, q_n, q_nm, q_n_var=0, q_nm_var=0,
                                  use_polynom=True):
        """ estimate the mutual information from given probabilities """
        Nr = self.Nr
        
        if use_polynom:
            # use the quadratic approximation of the mutual information
            MI = Nr - 0.5/LN2 * Nr * ((2*q_n - 1)**2 + 4*q_n_var)
            # add the effect of crosstalk
            MI -= 4/LN2 * Nr*(Nr - 1) * (q_nm**2 + q_nm_var)
            
        else:
            # use the better approximation based on the formula published in
            #     V. Sessak and R. Monasson, J Phys A, 42, 055001 (2009) 
            if not (np.isclose(q_n_var, 0) and np.isclose(q_nm_var, 0)):
                logging.warn('Estimating mutual information using the non-'
                             'polynomial form does not support the inclusion '
                             'of variances, yet.')
            
            # use exact expression for the entropy of uncorrelated receptors             
            MI = -Nr * (xlog2x(q_n) + xlog2x(1 - q_n))

            # add the effect of crosstalk
            if 0 < q_n < 1: 
                # TODO: add q_nm_var
                MI -= 0.5/LN2 * Nr*(Nr - 1)/2 * q_nm**2 / (q_n**2 - q_n)**2
        
        return MI
    
        
    def _estimate_MI_from_r_values(self, r_n, r_nm):
        """ estimate the mutual information from given probabilities """
        # calculate the crosstalk
        q_nm = r_nm - np.outer(r_n, r_n)
        return self._estimate_MI_from_q_values(r_n, q_nm)
      
        
    def _estimate_MI_from_r_stats(self, r_n, r_nm, r_n_var=0, r_nm_var=0,
                                  ret_var=False):
        """ estimate the mutual information from given probabilities """
        if r_nm_var != 0:
            raise NotImplementedError('Correlation calculations are not tested.')
        # calculate the crosstalk
        q_nm = r_nm - r_n**2 - r_n_var
        q_nm_var = r_nm_var + 4*r_n**2*r_n_var + 2*r_n_var**2
        return self._estimate_MI_from_q_stats(r_n, q_nm, r_n_var, q_nm_var,
                                              ret_var=ret_var)
      
    

class LibraryNumericMixin(object):
    """ Mixin class that defines functions that are useful for numerical 
    calculations.
    
    The iteration over mixtures must be implemented in the subclass. Here, we 
    expect the methods `_sample_mixtures` and `_sample_steps` to exist, which
    are a generator of mixtures and its expected length, respectively.    
    """
    
    def concentration_statistics(self, method='auto', **kwargs):
        """ calculates mixture statistics using a metropolis algorithm
        Returns the mean concentration, the variance, and the covariance matrix.

        `method` can be one of [monte_carlo', 'estimate'].
        """
        if method == 'auto':
            method = 'monte_carlo'
                
        if method == 'monte_carlo' or method == 'monte-carlo':
            return self.concenration_statistics_monte_carlo(**kwargs)
        elif method == 'estimate':
            return self.concentration_statistics_estimate(**kwargs)
        else:
            raise ValueError('Unknown method `%s`.' % method)
            
            
    def concentration_statistics_monte_carlo(self):
        """ calculates mixture statistics using a metropolis algorithm """
        count = 0
        hist1d = np.zeros(self.Ns)
        hist2d = np.zeros((self.Ns, self.Ns))

        # sample mixtures uniformly
        # FIXME: use better online algorithm that is less prone to canceling
        for c in self._sample_mixtures():
            count += 1
            hist1d += c
            hist2d += np.outer(c, c)
        
        # calculate the frequency and the correlations 
        ci_mean = hist1d/count
        cij_corr = hist2d/count - np.outer(ci_mean, ci_mean)
        
        ci_var = np.diag(cij_corr)
        return {'mean': ci_mean, 'std': np.sqrt(ci_var), 'var': ci_var,
                'cov': cij_corr}


    def excitation_statistics(self, method='auto', ret_correlations=True,
                              **kwargs):
        """ calculates the statistics of the excitation of the receptors.
        Returns the mean excitation, the variance, and the covariance matrix.

        `method` can be one of [monte_carlo', 'estimate'].
        """
        if method == 'auto':
            method = 'monte_carlo'
                
        if method == 'monte_carlo' or method == 'monte-carlo':
            return self.excitation_statistics_monte_carlo(ret_correlations,
                                                          **kwargs)
        elif method == 'estimate':
            return self.excitation_statistics_estimate(**kwargs)
        else:
            raise ValueError('Unknown method `%s`.' % method)
        
        
    def excitation_statistics_monte_carlo(self, ret_correlations=False):
        """
        calculates the statistics of the excitation of the receptors.
        Returns the mean excitation, the variance, and the covariance matrix.
        
        The algorithms used here have been taken from
            https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance
        """
        # prevent integer overflow in collecting activity patterns
        assert self.Nr <= self.parameters['max_num_receptors'] <= 63

        S_ni = self.sens_mat

        if ret_correlations:
            # calculate the mean and the covariance matrix
    
            # prepare variables holding the necessary data
            en_mean = np.zeros(self.Nr)
            enm_cov = np.zeros((self.Nr, self.Nr))
            
            # sample mixtures and safe the requested data
            for count, c_i in enumerate(self._sample_mixtures(), 1):
                e_n = np.dot(S_ni, c_i)
                delta = (e_n - en_mean) / count
                en_mean += delta
                enm_cov += (count - 1) * np.outer(delta, delta) - enm_cov / count
                
            # calculate the requested statistics
            if count < 2:
                enm_cov.fill(np.nan)
            else:
                enm_cov *= count / (count - 1)
            
            en_var = np.diag(enm_cov)
            
            return {'mean': en_mean, 'std': np.sqrt(en_var), 'var': en_var,
                    'cov': enm_cov}
            
        else:
            # only calculate the mean and the variance
    
            # prepare variables holding the necessary data
            en_mean = np.zeros(self.Nr)
            en_square = np.zeros(self.Nr)
            
            # sample mixtures and safe the requested data
            for count, c_i in enumerate(self._sample_mixtures(), 1):
                e_n = np.dot(S_ni, c_i)
                delta = e_n - en_mean
                en_mean += delta / count
                en_square += delta*(e_n - en_mean)
                
            # calculate the requested statistics
            if count < 2:
                en_var.fill(np.nan)
            else:
                en_var = en_square / (count - 1)
    
            return {'mean': en_mean, 'std': np.sqrt(en_var), 'var': en_var}
                            
    
    def excitation_statistics_estimate(self, **kwargs):
        """
        calculates the statistics of the excitation of the receptors.
        Returns the mean excitation, the variance, and the covariance matrix.
        """
        c_stats = self.concentration_statistics_estimate(**kwargs)
        
        # calculate statistics of e_n = \sum_i S_ni * c_i        
        S_ni = self.sens_mat
        en_mean = np.dot(S_ni, c_stats['mean'])
        cov_is_diagonal = c_stats.get('cov_is_diagonal', False)
        if cov_is_diagonal:
            enm_cov = np.einsum('ni,mi,i->nm', S_ni, S_ni, c_stats['var'])
        else:
            enm_cov = np.einsum('ni,mj,ij->nm', S_ni, S_ni, c_stats['cov'])
        en_var = np.diag(enm_cov)
        
        return {'mean': en_mean, 'std': np.sqrt(en_var), 'var': en_var,
                'cov': enm_cov}
        
        
    def receptor_activity(self, method='auto', ret_correlations=False, **kwargs):
        """ calculates the average activity of each receptor
        
        `method` can be one of [monte_carlo', 'estimate'].
        """
        if method == 'auto':
            method = 'monte_carlo'
                
        if method == 'monte_carlo' or method == 'monte-carlo':
            return self.receptor_activity_monte_carlo(ret_correlations, **kwargs)
        elif method == 'estimate':
            return self.receptor_activity_estimate(ret_correlations, **kwargs)
        else:
            raise ValueError('Unknown method `%s`.' % method)
     
            
    def receptor_activity_monte_carlo(self, ret_correlations=False):
        """ calculates the average activity of each receptor """
        # prevent integer overflow in collecting activity patterns
        assert self.Nr <= self.parameters['max_num_receptors'] <= 63

        S_ni = self.sens_mat

        r_n = np.zeros(self.Nr)
        if ret_correlations:
            r_nm = np.zeros((self.Nr, self.Nr))
        
        for c_i in self._sample_mixtures():
            e_n = np.dot(S_ni, c_i)
            a_n = (e_n >= 1)
            r_n[a_n] += 1
            if ret_correlations:
                r_nm[np.outer(a_n, a_n)] += 1
            
        r_n /= self._sample_steps
        if ret_correlations:
            r_nm /= self._sample_steps
            return r_n, r_nm
        else:
            return r_n        
         
                        
    def receptor_activity_estimate(self, ret_correlations=False,
                                   excitation_model=None, clip=False):
        """ estimates the average activity of each receptor """
        en_stats = self.excitation_statistics_estimate()

        # calculate the receptor activity
        r_n = self._estimate_qn_from_en(en_stats,
                                        excitation_model=excitation_model)
        if clip:
            np.clip(r_n, 0, 1, r_n)

        if ret_correlations:
            # calculate the correlated activity 
            q_nm = self._estimate_qnm_from_en(en_stats)
            r_nm = np.outer(r_n, r_n) + q_nm
            if clip:
                np.clip(r_nm, 0, 1, r_nm)

            return r_n, r_nm
        else:
            return r_n   
        

    def receptor_crosstalk(self, method='auto', ret_receptor_activity=False,
                           clip=False, **kwargs):
        """ calculates the average activity of the receptor as a response to 
        single ligands.
        
        `method` can be ['brute_force', 'monte_carlo', 'estimate', 'auto'].
            If it is 'auto' than the method is chosen automatically based on the
            problem size.
        """
        if method == 'estimate':
            kwargs['clip'] = False

        # calculate receptor activities with the requested `method`            
        r_n, r_nm = self.receptor_activity(method, ret_correlations=True,
                                           **kwargs)
        
        # calculate receptor crosstalk from the observed probabilities
        q_nm = r_nm - np.outer(r_n, r_n)
        if clip:
            np.clip(q_nm, 0, 1, q_nm)
        
        if ret_receptor_activity:
            return r_n, q_nm # q_n = r_n
        else:
            return q_nm

        
    def receptor_crosstalk_estimate(self, ret_receptor_activity=False,
                                    excitation_model=None, clip=False):
        """ calculates the average activity of the receptor as a response to 
        single ligands. """
        en_stats = self.excitation_statistics_estimate()

        # calculate the receptor crosstalk
        q_nm = self._estimate_qnm_from_en(en_stats)
        if clip:
            np.clip(q_nm, 0, 1, q_nm)

        if ret_receptor_activity:
            # calculate the receptor activity
            q_n = self._estimate_qn_from_en(en_stats, excitation_model)
            if clip:
                np.clip(q_n, 0, 1, q_n)

            return q_n, q_nm
        else:
            return q_nm        
        
        
    def mutual_information(self, method='auto', ret_prob_activity=False,
                           **kwargs):
        """ calculate the mutual information of the receptor array.

        `method` can be one of [monte_carlo', 'estimate'].
        """
        if method == 'auto':
            method = 'monte_carlo'
                
        if method == 'monte_carlo' or method == 'monte-carlo':
            return self.mutual_information_monte_carlo(ret_prob_activity)
        elif method == 'estimate':
            return self.mutual_information_estimate(ret_prob_activity, **kwargs)
        else:
            raise ValueError('Unknown method `%s`.' % method)

                                                   
    def mutual_information_monte_carlo(self, ret_prob_activity=False):
        """ calculate the mutual information using a monte carlo strategy. The
        number of steps is given by the model parameter 'monte_carlo_steps' """
        # prevent integer overflow in collecting activity patterns
        assert self.Nr <= self.parameters['max_num_receptors'] <= 63

        base = 2 ** np.arange(0, self.Nr)

        # sample mixtures according to the probabilities of finding
        # substrates
        count_a = np.zeros(2**self.Nr)
        for c in self._sample_mixtures():
            # get the activity vector ...
            a = (np.dot(self.sens_mat, c) >= 1)
            
            # ... and represent it as a single integer
            a_id = np.dot(base, a)
            # increment counter for this output
            count_a[a_id] += 1
            
        # count_a contains the number of times output pattern a was observed.
        # We can thus construct P_a(a) from count_a. 
        q_n = count_a / count_a.sum()
        
        # calculate the mutual information from the result pattern
        MI = -sum(q*np.log2(q) for q in q_n if q != 0)

        if ret_prob_activity:
            return MI, q_n
        else:
            return MI

                    
    def mutual_information_estimate(self, ret_prob_activity=False,
                                    excitation_model='default', clip=True,
                                    use_polynom=True):
        """ returns a simple estimate of the mutual information.
        `clip` determines whether the approximated probabilities should be
            clipped to [0, 1] before being used to calculate the mutual info.
        """
        q_n, q_nm = self.receptor_crosstalk_estimate(
            ret_receptor_activity=True, \
            excitation_model=excitation_model,
            clip=clip
        )
        
        # calculate the approximate mutual information
        MI = self._estimate_MI_from_q_values(q_n, q_nm, use_polynom=use_polynom)
        
        if clip:
            MI = np.clip(MI, 0, self.Nr)
        
        if ret_prob_activity:
            return MI, q_n
        else:
            return MI
        
        

@vectorize_double
def _estimate_qn_from_en_lognorm(en_mean, en_var):
    """ estimates probability q_n that a receptor is activated by a mixture
    based on the statistics of the excitations s_n assuming an underlying
    log-normal distribution for s_n """
    if en_mean == 0:
        q_n = 0.
    elif np.isclose(en_var, 0):
        q_n = np.double(en_mean > 1)
    else:
        en_cv2 = en_var / en_mean**2
        enum = np.log(np.sqrt(1 + en_cv2) / en_mean)
        denom = np.sqrt(2*np.log(1 + en_cv2))
        q_n = 0.5 * special.erfc(enum/denom)
        
    return q_n
       
               

@vectorize_double
def _estimate_qn_from_en_lognorm_approx(en_mean, en_var):
    """ estimates probability q_n that a receptor is activated by a mixture
    based on the statistics of the excitations s_n using an approximation """
    if np.isclose(en_var, 0):
        q_n = np.double(en_mean > 1)
    else:                
        q_n = (0.5
               + (en_mean - 1) / np.sqrt(2*np.pi*en_var)
               + (5*en_mean - 7) * np.sqrt(en_var/(32*np.pi))
               )
        # here, the last term comes from an expansion of the log-normal approx.

    return np.clip(q_n, 0, 1)



@vectorize_double
def _estimate_qn_from_en_gaussian(en_mean, en_var):
    """ estimates probability q_n that a receptor is activated by a mixture
    based on the statistics of the excitations s_n assuming an underlying
    normal distribution for s_n """
    if np.isclose(en_var, 0):
        q_n = np.double(en_mean > 1)
    else:
        q_n = 0.5 * special.erfc((1 - en_mean)/np.sqrt(2 * en_var))
        
    return q_n
           
               

@vectorize_double
def _estimate_qn_from_en_gaussian_approx(en_mean, en_var):
    """ estimates probability q_n that a receptor is activated by a mixture
    based on the statistics of the excitations s_n using an approximation """
    if np.isclose(en_var, 0):
        q_n = np.double(en_mean > 1)
    else:                
        q_n = 0.5 + (en_mean - 1) / np.sqrt(2*np.pi*en_var)

    return np.clip(q_n, 0, 1)



def _ensemble_average_job(args):
    """ helper function for calculating ensemble averages using
    multiprocessing """
    # We have to initialize the random number generator for each process
    # because we would have the same random sequence for all processes
    # otherwise.
    np.random.seed()
    
    # create the object ...
    obj = args[0](**args[1])
    # ... and evaluate the requested method
    if len(args) > 2: 
        return getattr(obj, args[2])(**args[3])
    else:
        return getattr(obj, args[2])()

    
    