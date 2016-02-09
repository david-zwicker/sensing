* Think about estimating/calculating mixture entropy for continuous mixtures
* Think about implementing the sped up version of the mutual information
    calculation for correlated mixtures by representing the concentration
    vector by the indices of the ligands that are present
* Introduce statistics class, that calculates std, cov on the fly and does not
    use memory for this
    - this should behave as a dictionary to be drop-in replacement