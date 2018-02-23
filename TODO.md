# TODO

This file collects bugs and feature requests

## High priority

## Low priority
* Prevent numba warnings to be emitted too often
    - Using warnings module to emit numba warnings? 
* Use more efficient storage scheme for mutual information in primacy coding
    - it should be possible to find a mapping from activities with Nc on-bits
      to integers in the range 1 ... binom(Nr, Nc). Currently, we use an array
      of 2**Nr items, where most activities are never seen
    - such a function is defined in https://stackoverflow.com/a/3143594/932593
      (the `unchoose` function is defined recursively, though)