from .lib_exp_numeric import LibraryExponentialNumeric
from .lib_exp_theory import LibraryExponentialLogNormal

__all__ = ['LibraryExponentialNumeric', 'LibraryExponentialLogNormal']

# try importing numba for speeding up calculations
try:
    from .numba_speedup import numba_patcher
    numba_patcher.enable() #< enable the speed-up by default
except ImportError:
    import logging
    logging.warning('Numba could not be loaded. Slow functions will be used')
    numba_patcher = None
