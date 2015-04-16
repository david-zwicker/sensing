#!/usr/bin/env python
'''
Created on Apr 16, 2015

@author: David Zwicker <dzwicker@seas.harvard.edu>
'''

from __future__ import division

import sys
import os.path
# append base path to sys.path
script_path = os.path.split(os.path.realpath(__file__))[0]
sys.path.append(os.path.join(script_path, '..', '..'))

import cProfile
import pstats
import tempfile

from binary_response import ReceptorLibraryNumeric

# create an object and trigger the numba just-in-time compilation
model = ReceptorLibraryNumeric.create_test_instance(Ns=6, Nr=3)
model.mutual_information_brute_force()

# run the profiler and save result to temporary file
cmd = "model.optimize_library('mutual_information', method='anneal', steps=10000)"
with tempfile.NamedTemporaryFile() as tmpfile:
    cProfile.run(cmd, tmpfile.name)
    stats = pstats.Stats(tmpfile.name)

# display the results
stats.strip_dirs()
stats.sort_stats('cumulative')
stats.print_stats(30)


