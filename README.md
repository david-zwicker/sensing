# Sensing #

This repository contains code for simulations of the sensing project.

Necessary python packages:

Package     | Usage                                      
------------|-------------------------------------------
numpy       | Array library used for manipulating data
scipy       | Miscellaneous scientific functions
simanneal   | Simulated annealing algorithm published on github


Optional python packages, which can be installed through `pip`:

Package      | Usage                                      
-------------|-------------------------------------------
numba        | For creating compiled code for faster processing


The classes are organized as follows:
- We distinguish between the case of binary mixtures (where a substrate is either
    present or not) and the case of continuous mixtures (where substrates are
    present at different concentrations). These are organized in different
    modules.
- We distinguish between general classes and classes  with a concrete receptor
    library (for numerical or analytical calculations)
