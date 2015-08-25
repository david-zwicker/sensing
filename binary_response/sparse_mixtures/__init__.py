from .library_numeric import LibrarySparseNumeric
from .library_theory import (LibrarySparseBinary, LibrarySparseLogNormal,
                             LibrarySparseLogUniform)

__all__ = ['LibrarySparseNumeric', 'LibrarySparseBinary',
           'LibrarySparseLogNormal', 'LibrarySparseLogUniform']

# try importing numba for speeding up calculations
try:
    from .numba_speedup import numba_patcher
    numba_patcher.enable() #< enable the speed-up by default
except ImportError:
    import logging
    logging.warn('Numba could not be loaded. Slow functions will be used')
    numba_patcher = None
