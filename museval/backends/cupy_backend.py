"""CuPy backend implementation for GPU computation.

This module wraps CuPy functions to implement the Backend interface.
"""
import os

try:
    import cupy as cp
    import cupyx.scipy.fftpack
    import cupyx.scipy.linalg
    import cupyx.scipy.signal
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False
    cp = None

from .base import Backend


class CupyBackend(Backend):
    """CuPy-based backend for GPU computation.
    
    This backend uses CuPy for GPU-accelerated computations. Requires CUDA
    and CuPy to be installed.
    
    Raises
    ------
    ImportError
        If CuPy is not installed.
    RuntimeError
        If CUDA is not available or no GPU devices are found.
    """
    
    def __init__(self):
        if not CUPY_AVAILABLE:
            raise ImportError(
                "CuPy is not installed. Install with: pip install museval[gpu]"
            )
        
        # Check CUDA availability
        if not cp.cuda.is_available():
            raise RuntimeError("CUDA is not available on this system")
        
        # Set device if specified in environment
        device_id = os.environ.get('MUSEVAL_GPU_DEVICE', None)
        if device_id is not None:
            try:
                cp.cuda.Device(int(device_id)).use()
            except Exception as e:
                raise RuntimeError(
                    f"Could not set GPU device {device_id}: {e}"
                )
    
    @property
    def name(self):
        return 'cupy'
    
    @property
    def array_type(self):
        return cp.ndarray
    
    # Array creation
    def asarray(self, arr):
        return cp.asarray(arr)
    
    def zeros(self, shape, dtype=None):
        return cp.zeros(shape, dtype=dtype)
    
    def empty(self, shape, dtype=None):
        return cp.empty(shape, dtype=dtype)
    
    def arange(self, *args, **kwargs):
        return cp.arange(*args, **kwargs)
    
    def array(self, obj, dtype=None):
        return cp.array(obj, dtype=dtype)
    
    def atleast_3d(self, arr):
        return cp.atleast_3d(arr)
    
    # Array manipulation
    def moveaxis(self, arr, source, destination):
        return cp.moveaxis(arr, source, destination)
    
    def reshape(self, arr, shape):
        return cp.reshape(arr, shape)
    
    def hstack(self, arrays):
        return cp.hstack(arrays)
    
    # Mathematical operations
    def sum(self, arr, axis=None, keepdims=False):
        return cp.sum(arr, axis=axis, keepdims=keepdims)
    
    def all(self, arr, axis=None):
        return cp.all(arr, axis=axis)
    
    def any(self, arr, axis=None):
        return cp.any(arr, axis=axis)
    
    def argmax(self, arr, axis=None):
        return cp.argmax(arr, axis=axis)
    
    def mean(self, arr, axis=None):
        return cp.mean(arr, axis=axis)
    
    def real(self, arr):
        return cp.real(arr)
    
    def conj(self, arr):
        return cp.conj(arr)
    
    def log10(self, arr):
        return cp.log10(arr)
    
    def ceil(self, arr):
        return cp.ceil(arr)
    
    def floor(self, arr):
        return cp.floor(arr)
    
    def min(self, arr, axis=None):
        return cp.min(arr, axis=axis)
    
    def isnan(self, arr):
        return cp.isnan(arr)
    
    def isinf(self, arr):
        return cp.isinf(arr)
    
    # FFT operations
    def fft(self, arr, n=None, axis=-1):
        return cupyx.scipy.fftpack.fft(arr, n=n, axis=axis)
    
    def ifft(self, arr, n=None, axis=-1):
        return cupyx.scipy.fftpack.ifft(arr, n=n, axis=axis)
    
    # Signal processing
    def fftconvolve(self, in1, in2, mode='full'):
        return cupyx.scipy.signal.fftconvolve(in1, in2, mode=mode)
    
    def toeplitz(self, c, r=None):
        return cupyx.scipy.linalg.toeplitz(c, r=r)
    
    # Linear algebra
    def solve(self, a, b):
        return cp.linalg.solve(a, b)
    
    def lstsq(self, a, b):
        return cp.linalg.lstsq(a, b, rcond=None)
    
    def eye(self, n, dtype=None):
        return cp.eye(n, dtype=dtype)
    
    # Array properties
    def asnumpy(self, arr):
        """Convert CuPy array to NumPy array (transfers from GPU to CPU)."""
        if isinstance(arr, cp.ndarray):
            return cp.asnumpy(arr)
        return arr
    
    def get_device(self):
        device = cp.cuda.Device()
        try:
            device_name = device.attributes.get('Name', b'Unknown GPU')
            if isinstance(device_name, bytes):
                device_name = device_name.decode()
        except (KeyError, AttributeError):
            device_name = 'Unknown GPU'
        return {
            'type': 'cuda',
            'id': device.id,
            'name': device_name,
            'compute_capability': device.compute_capability
        }
    
    @property
    def inf(self):
        return cp.inf
    
    @property
    def nan(self):
        return cp.nan
    
    def finfo(self, dtype):
        return cp.finfo(dtype)

