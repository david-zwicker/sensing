from .pc_numeric import PrimacyCodingNumeric
from .pc_theory import PrimacyCodingTheory

__all__ = ['PrimacyCodingNumeric', 'PrimacyCodingTheory']


# try importing numba for speeding up calculations for numeric module
try:
    from .numba_speedup_numeric import numba_patcher as number_patcher_numeric
except ImportError:
    # numba does not seem to be available -> fall back on python methods
    import logging
    logging.warning('Numba could not be loaded. Slow functions will be used')
    number_patcher_numeric = None
else:
    # enable the speed-up by default
    number_patcher_numeric.enable()


# try importing numba for speeding up calculations for theory module
try:
    from .numba_speedup_numeric import numba_patcher as number_patcher_theory
except ImportError:
    # numba does not seem to be available -> fall back on python methods
    import logging
    logging.warning('Numba could not be loaded. Slow functions will be used')
    number_patcher_theory = None
else:
    # enable the speed-up by default
    number_patcher_theory.enable()
