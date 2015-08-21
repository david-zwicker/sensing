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
    numba_patcher = None
    print('Numba patches could not be applied. Slow functions will be used')