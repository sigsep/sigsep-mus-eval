"""NumPy backend implementation for CPU computation.

This module wraps NumPy and SciPy functions to implement the Backend interface.
"""
import numpy as np
import scipy.fftpack
from scipy.linalg import toeplitz
from scipy.signal import fftconvolve

from .base import Backend


class NumpyBackend(Backend):
    """NumPy-based backend for CPU computation.
    
    This backend uses NumPy and SciPy for all computations on the CPU.
    """
    
    @property
    def name(self):
        return 'numpy'
    
    @property
    def array_type(self):
        return np.ndarray
    
    # Array creation
    def asarray(self, arr):
        return np.asarray(arr)
    
    def zeros(self, shape, dtype=None):
        return np.zeros(shape, dtype=dtype)
    
    def empty(self, shape, dtype=None):
        return np.empty(shape, dtype=dtype)
    
    def arange(self, *args, **kwargs):
        return np.arange(*args, **kwargs)
    
    def array(self, obj, dtype=None):
        return np.array(obj, dtype=dtype)
    
    def atleast_3d(self, arr):
        return np.atleast_3d(arr)
    
    # Array manipulation
    def moveaxis(self, arr, source, destination):
        return np.moveaxis(arr, source, destination)
    
    def reshape(self, arr, shape):
        return np.reshape(arr, shape)
    
    def hstack(self, arrays):
        return np.hstack(arrays)
    
    # Mathematical operations
    def sum(self, arr, axis=None, keepdims=False):
        return np.sum(arr, axis=axis, keepdims=keepdims)
    
    def all(self, arr, axis=None):
        return np.all(arr, axis=axis)
    
    def any(self, arr, axis=None):
        return np.any(arr, axis=axis)
    
    def argmax(self, arr, axis=None):
        return np.argmax(arr, axis=axis)
    
    def mean(self, arr, axis=None):
        return np.mean(arr, axis=axis)
    
    def real(self, arr):
        return np.real(arr)
    
    def conj(self, arr):
        return np.conj(arr)
    
    def log10(self, arr):
        return np.log10(arr)
    
    def ceil(self, arr):
        return np.ceil(arr)
    
    def floor(self, arr):
        return np.floor(arr)
    
    def min(self, arr, axis=None):
        return np.min(arr, axis=axis)
    
    def isnan(self, arr):
        return np.isnan(arr)
    
    def isinf(self, arr):
        return np.isinf(arr)
    
    # FFT operations
    def fft(self, arr, n=None, axis=-1):
        return scipy.fftpack.fft(arr, n=n, axis=axis)
    
    def ifft(self, arr, n=None, axis=-1):
        return scipy.fftpack.ifft(arr, n=n, axis=axis)
    
    # Signal processing
    def fftconvolve(self, in1, in2, mode='full'):
        return fftconvolve(in1, in2, mode=mode)
    
    def toeplitz(self, c, r=None):
        return toeplitz(c, r=r)
    
    # Linear algebra
    def solve(self, a, b):
        return np.linalg.solve(a, b)
    
    def lstsq(self, a, b):
        return np.linalg.lstsq(a, b, rcond=None)
    
    def eye(self, n, dtype=None):
        return np.eye(n, dtype=dtype)
    
    # Array properties
    def asnumpy(self, arr):
        """Already NumPy array, return as-is."""
        return np.asarray(arr)
    
    def get_device(self):
        return {'type': 'cpu', 'name': 'CPU'}
    
    # Constants
    @property
    def inf(self):
        return np.inf
    
    @property
    def nan(self):
        return np.nan
    
    def finfo(self, dtype):
        return np.finfo(dtype)

