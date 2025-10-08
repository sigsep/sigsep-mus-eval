"""Backend abstraction layer for CPU and GPU computation.

This module provides a unified interface for switching between NumPy (CPU)
and CuPy (GPU) backends for BSS evaluation metrics computation.
"""
import os
import warnings


def get_backend(backend='auto'):
    """Select and return the appropriate computation backend.
    
    Parameters
    ----------
    backend : str, optional
        Backend to use: 'numpy' (CPU), 'cupy' (GPU), or 'auto'.
        Default is 'auto', which uses the MUSEVAL_BACKEND environment
        variable or falls back to 'numpy'.
        
    Returns
    -------
    backend : Backend
        An instance of the selected backend implementing the Backend interface.
        
    Raises
    ------
    ValueError
        If an unknown backend name is provided.
        
    Examples
    --------
    >>> backend = get_backend('numpy')
    >>> arr = backend.zeros((10, 10))
    >>> backend = get_backend('cupy')  # Requires CuPy installation
    """
    if backend == 'auto':
        backend = os.environ.get('MUSEVAL_BACKEND', 'numpy')
    
    backend = backend.lower()
    
    if backend == 'numpy':
        from .numpy_backend import NumpyBackend
        return NumpyBackend()
    elif backend == 'cupy':
        try:
            from .cupy_backend import CupyBackend
            return CupyBackend()
        except ImportError as e:
            warnings.warn(
                f"CuPy not available ({e}). Install with: "
                "pip install museval[gpu]. Falling back to NumPy.",
                UserWarning
            )
            from .numpy_backend import NumpyBackend
            return NumpyBackend()
        except RuntimeError as e:
            warnings.warn(
                f"CUDA not available ({e}). Falling back to NumPy.",
                UserWarning
            )
            from .numpy_backend import NumpyBackend
            return NumpyBackend()
    else:
        raise ValueError(
            f"Unknown backend: '{backend}'. "
            "Valid options are: 'numpy', 'cupy', 'auto'"
        )


__all__ = ['get_backend']

