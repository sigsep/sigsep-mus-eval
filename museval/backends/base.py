"""Base backend interface definition.

This module defines the abstract interface that all backends must implement.
"""
from abc import ABC, abstractmethod


class Backend(ABC):
    """Abstract base class for computation backends.
    
    All backends (NumPy, CuPy, etc.) must implement this interface to ensure
    compatibility with the metrics computation code.
    """
    
    @property
    @abstractmethod
    def name(self):
        """Return the name of the backend."""
        pass
    
    @property
    @abstractmethod
    def array_type(self):
        """Return the array type used by this backend."""
        pass
    
    # Array creation
    @abstractmethod
    def asarray(self, arr):
        """Convert array-like to backend array."""
        pass
    
    @abstractmethod
    def zeros(self, shape, dtype=None):
        """Create array filled with zeros."""
        pass
    
    @abstractmethod
    def empty(self, shape, dtype=None):
        """Create uninitialized array."""
        pass
    
    @abstractmethod
    def arange(self, *args, **kwargs):
        """Create array with evenly spaced values."""
        pass
    
    @abstractmethod
    def array(self, obj, dtype=None):
        """Create array from sequence."""
        pass
    
    @abstractmethod
    def atleast_3d(self, arr):
        """View inputs as arrays with at least three dimensions."""
        pass
    
    # Array manipulation
    @abstractmethod
    def moveaxis(self, arr, source, destination):
        """Move axes of an array to new positions."""
        pass
    
    @abstractmethod
    def reshape(self, arr, shape):
        """Reshape array to new shape."""
        pass
    
    @abstractmethod
    def hstack(self, arrays):
        """Stack arrays in sequence horizontally."""
        pass
    
    # Mathematical operations
    @abstractmethod
    def sum(self, arr, axis=None, keepdims=False):
        """Sum of array elements."""
        pass
    
    @abstractmethod
    def all(self, arr, axis=None):
        """Test whether all array elements evaluate to True."""
        pass
    
    @abstractmethod
    def any(self, arr, axis=None):
        """Test whether any array element evaluates to True."""
        pass
    
    @abstractmethod
    def argmax(self, arr, axis=None):
        """Return indices of maximum values."""
        pass
    
    @abstractmethod
    def mean(self, arr, axis=None):
        """Compute arithmetic mean."""
        pass
    
    @abstractmethod
    def real(self, arr):
        """Return real part of complex array."""
        pass
    
    @abstractmethod
    def conj(self, arr):
        """Return complex conjugate."""
        pass
    
    @abstractmethod
    def log10(self, arr):
        """Return base-10 logarithm."""
        pass
    
    @abstractmethod
    def ceil(self, arr):
        """Return ceiling of the input."""
        pass
    
    @abstractmethod
    def floor(self, arr):
        """Return floor of the input."""
        pass
    
    @abstractmethod
    def min(self, arr, axis=None):
        """Return minimum along axis."""
        pass
    
    @abstractmethod
    def isnan(self, arr):
        """Test element-wise for NaN."""
        pass
    
    @abstractmethod
    def isinf(self, arr):
        """Test element-wise for infinity."""
        pass
    
    # FFT operations
    @abstractmethod
    def fft(self, arr, n=None, axis=-1):
        """Compute one-dimensional FFT."""
        pass
    
    @abstractmethod
    def ifft(self, arr, n=None, axis=-1):
        """Compute one-dimensional inverse FFT."""
        pass
    
    # Signal processing
    @abstractmethod
    def fftconvolve(self, in1, in2, mode='full'):
        """Convolve two N-dimensional arrays using FFT."""
        pass
    
    @abstractmethod
    def toeplitz(self, c, r=None):
        """Construct a Toeplitz matrix."""
        pass
    
    # Linear algebra
    @abstractmethod
    def solve(self, a, b):
        """Solve linear equation a x = b."""
        pass
    
    @abstractmethod
    def lstsq(self, a, b):
        """Return least-squares solution to a x = b."""
        pass
    
    @abstractmethod
    def eye(self, n, dtype=None):
        """Return identity matrix."""
        pass
    
    # Array properties
    @abstractmethod
    def asnumpy(self, arr):
        """Convert backend array to NumPy array (CPU memory)."""
        pass
    
    @abstractmethod
    def get_device(self):
        """Get current device information."""
        pass
    
    # Constants
    @property
    @abstractmethod
    def inf(self):
        """Positive infinity constant."""
        pass
    
    @property
    @abstractmethod
    def nan(self):
        """NaN (Not a Number) constant."""
        pass
    
    @abstractmethod
    def finfo(self, dtype):
        """Machine limits for floating point types."""
        pass

