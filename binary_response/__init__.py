from model_numeric import ReceptorLibraryNumeric
from model_theory import ReceptorLibraryUniform

# try importing numba for speeding up calculations
try:
    from .numba_speedup import NumbaPatcher
    NumbaPatcher.enable() #< enable the speed-up by default
except ImportError:
    NumbaPatcher = None
    print('NumbaPatcher could not be loaded. Slow functions will be used')