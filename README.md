# Sensing

This repository contains code for studying the information processing of an
array of sensors, in particular in the olfactory system.

This package is the basis of the following three published papers:
* D. Zwicker, A. Murugan, and M. P. Brenner,
  "Receptor arrays optimized for natural odor statistics",
  [*Proc. Natl. Acad. Sci. USA* 113 (2016)](http://dx.doi.org/10.1073/pnas.1600357113)
* D. Zwicker, 
  "Normalized Neural Representations of Complex Odors",
  [*PLoS One* 11 (2016)](http://dx.doi.org/10.1371/journal.pone.0166456)
* D. Zwicker,
  "Primacy coding facilitates effective odor discrimination when receptor sensitivities are tuned",
  (under review)


An example of how to use this package can be found in the github repository
[sensing-normalized-results](https://github.com/david-zwicker/sensing-normalized-results),
which provides the scripts for generating the figures of the middle paper above.


## Installation

The code in this package should run under python 2.7 and python 3.x.
However, not all combinations of python versions and package versions have been
tested.
To obtain a first idea whether the package is functional, run `python -m unittest discover`
in the root directory of the package, which runs some unittests.
Note that thhis may take a while and might also yield errors due to numerical
randomness. 

The package only requires a few necessary python packages, which should be
available through `macports`, `pip`, or your system's package manager:

Package     | Usage                                      
------------|-------------------------------------------
numpy       | Array library used for manipulating data
scipy       | Miscellaneous scientific functions
six         | Compatibility layer to support python 2 and 3
utils       | Utility functions published at https://github.com/david-zwicker/py-utils

Optional python packages, which can be installed through `pip` or similar means
include:

Package     | Usage                                      
------------|-------------------------------------------
coverage    | For measuring test coverage
cma         | For optimizations using CMA-ES
numba       | For creating compiled code for faster processing
nose        | For parallel testing
simanneal   | Simulated annealing algorithm published on github


## Project structure

The classes in the project are organized as follows:
- We distinguish several different odor mixtures as inputs:
    - binary mixtures: ligands are either present or absent. The probability
        of ligands is controlled by an Ising-like distribution.
    - continuous mixtures: all ligands are present at random concentrations. The
        probability distribution of the concentration vector is specified by the
        mean concentrations and a covariance matrix. Exponential mixtures and
        gaussian mixtures have been implemented.
    - sparse mixtures: the ligands that are present have random concentrations.
        Here, we use the algorithm from the binary mixtures to determine which
        ligands are present in a mixture and then chose their concentrations
        independently from exponential distributions. 
    The code for these mixtures is organized in sub-modules in the
    `binary_response` module.
- The classes in `binary_response` all implement a simple binary representation,
  where a fixed threshold is used.
- Two adaptive thresholds are implemented in the separate modules `adaptive_threshold`
  and `primacy_coding`. They extend the code in `binary_response`.
- We distinguish between general classes and classes  with a concrete receptor
    library. Here, we distinguish libraries that do numerical simulations and
    libraries that provide analytical results.
