from .library_numeric import LibraryBinaryNumeric
from .library_theory import LibraryBinaryUniform

# provide deprecated classes for compatibility
from utils.misc import DeprecationHelper
ReceptorLibraryNumeric = DeprecationHelper(LibraryBinaryNumeric)
ReceptorLibraryUniform = DeprecationHelper(LibraryBinaryUniform)

__all__ = ['LibraryBinaryNumeric', 'LibraryBinaryUniform',
           'ReceptorLibraryNumeric', 'ReceptorLibraryUniform']

# try importing numba for speeding up calculations
try:
    from .numba_speedup import numba_patcher
    numba_patcher.enable() #< enable the speed-up by default
except ImportError:
    numba_patcher = None
    print('Numba patches could not be applied. Slow functions will be used')