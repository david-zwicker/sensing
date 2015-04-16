* Add code to estimate the mutual information for large systems by using a monte
    carlo method instead of brute force
* Implement the monte carlo method on the GPU, since it should be really scalable
    - the cuda random number library is not available with the standard numba
    - we could write everything in C, but that would have to include the
        simulated annealing as well, since the calculation of the mutual
        information is not the slow part 