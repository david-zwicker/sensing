'''
Created on May 14, 2015

@author: David Zwicker <dzwicker@seas.harvard.edu>
'''

from __future__ import division

import functools
import random
import time

from simanneal import Annealer


   
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

