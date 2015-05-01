from libraries.binary_numeric import LibraryBinaryNumeric
from libraries.binary_theory import LibraryBinaryUniform


# provide deprecated classes for compatibility
from .utils import DeprecationHelper
ReceptorLibraryUniform = DeprecationHelper(LibraryBinaryUniform)
ReceptorLibraryNumeric = DeprecationHelper(LibraryBinaryNumeric)