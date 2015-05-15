#!/usr/bin/env python
'''
Created on Apr 3, 2015

@author: David Zwicker <dzwicker@seas.harvard.edu>
'''

from __future__ import division

import sys
import os.path
# append base path to sys.path
script_path = os.path.split(os.path.realpath(__file__))[0]
sys.path.append(os.path.join(script_path, '..', '..'))

import argparse
import itertools
import multiprocessing as mp

import six.moves.cPickle as pickle

from binary_response import LibraryBinaryNumeric, LibraryBinaryUniform



def optimize_receptors(parameters):
    """ optimize receptors of the system described by `parameters` """
    # get an estimate of the optimal response fraction
    theory = LibraryBinaryUniform(parameters['Ns'], parameters['Nr'])
    theory.set_commonness('const', parameters['d'])
    density_optimal = theory.density_optimal()
    
    # setup the numerical model that we use for optimization
    model = LibraryBinaryNumeric(
        parameters['Ns'], parameters['Nr'],
        parameters={'verbosity': 0 if parameters['quite'] else 1,
                    'random_seed': parameters['random_seed'],}
    )
    model.choose_interaction_matrix(density=density_optimal)
    model.set_commonness(parameters['scheme'], parameters['d'])
    
    # optimize
    result = model.optimize_library('mutual_information', method='anneal',
                                    steps=parameters['steps'])
    
    return {'parameters': parameters, 'init_arguments': model.init_arguments,
            'MI': result[0], 'I_ai': result[1]}



def main():
    """ main program """
    
    # setup the argument parsing
    parser = argparse.ArgumentParser(
         description='Program to optimize receptors for the given parameters. '
                     'Note that most parameters can take multiple values, in '
                     'which case all parameter combinations are computed.',
         formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('-Ns', nargs='+', type=int, required=True,
                        default=argparse.SUPPRESS, help='number of substrates')
    parser.add_argument('-Nr', nargs='+', type=int, required=True,
                        default=argparse.SUPPRESS, help='number of receptors')
    parser.add_argument('-d', nargs='+', type=float, required=True,
                        default=argparse.SUPPRESS,
                        help='average number of substrates per mixture')
    parser.add_argument('-steps', '-s', nargs='+', type=int, default=[100000],
                        help='steps in simulated annealing')
    parser.add_argument('-repeat', '-r', type=int, default=1,
                        help='number of repeats for each parameter set')
    parser.add_argument('-scheme', type=str, default='random_uniform',
                        choices=['const', 'linear', 'random_uniform'],
                        help='scheme for picking substrate probabilities')
    parser.add_argument('-seed', type=int, default=None,
                        help='seed for the random number generator.')
    parser.add_argument('-parallel', '-p', action='store_true',
                        default=False, help='use multiple processes')
    parser.add_argument('-quite', '-q', action='store_true',
                        default=False,
                        help='silence the output')
    parser.add_argument('-filename', '-f', default='result.pkl',
                        help='filename of the result file')
    
    # fetch the arguments and build the parameter list
    args = parser.parse_args()
    arg_list = (args.Ns, args.Nr, args.d, args.steps, range(args.repeat))
    parameter_list = [{'Ns': Ns, 'Nr': Nr, 'd': d, 'steps': steps,
                       'scheme': args.scheme, 'random_seed': args.seed,
                       'quite': args.quite}
                      for Ns, Nr, d, steps, _ in itertools.product(*arg_list)]
        
    # do the optimization
    if args.parallel and len(parameter_list) > 1:
        results = mp.Pool().map(optimize_receptors, parameter_list)
    else:
        results = map(optimize_receptors, parameter_list)
        
    # write the pickled result to file
    with open(args.filename, 'wb') as fp:
        pickle.dump(results, fp, pickle.HIGHEST_PROTOCOL)
    
    

if __name__ == '__main__':
    main()
    
