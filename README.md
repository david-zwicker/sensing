# Sensing #

This repository contains code for simulations of the sensing project.

Necessary python packages:

Package     | Usage                                      
------------|-------------------------------------------
numpy       | Array library used for manipulating data
scipy       | Miscellaneous scientific functions
six         | Compatibility layer to support python 2 and 3


Optional python packages, which can be installed through `pip`:

Package     | Usage                                      
------------|-------------------------------------------
coverage    | For measuring test coverage
numba       | For creating compiled code for faster processing
nose        | For parallel testing
simanneal   | Simulated annealing algorithm published on github


The classes in the project are organized as follows:
- We distinguish several different odor mixtures as inputs:
    - binary mixtures: ligands are either present or absent
    - continuous mixtures: all ligands are present at random concentrations
    - sparse mixtures: the ligands that are present have random concentrations
    The code for these mixtures is organized in different modules.
- We distinguish between general classes and classes  with a concrete receptor
    library. Here, we distinguish libraries that do numerical simulations and
    librareis that provide analytical results.
