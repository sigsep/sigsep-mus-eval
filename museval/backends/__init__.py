"""Backend abstraction layer for CPU and GPU computation.

This module provides a unified interface for switching between NumPy (CPU)
and CuPy (GPU) backends for BSS evaluation metrics computation.
"""

import os
import warnings


def _detect_best_backend():
    """Detect the best available backend (GPU first, then CPU).

    Returns
    -------
    str
        'cupy' if GPU is available, otherwise 'numpy'.
    """
    try:
        import cupy as cp

        if cp.cuda.is_available():
            return "cupy"
    except (ImportError, RuntimeError):
        pass
    return "numpy"


def get_backend(backend="auto"):
    """Select and return the appropriate computation backend.

    Parameters
    ----------
    backend : str, optional
        Backend to use: 'numpy' (CPU), 'cupy' (GPU), or 'auto'.
        Default is 'auto', which attempts to use GPU if available,
        otherwise falls back to CPU. The MUSEVAL_BACKEND environment
        variable can override this behavior.

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
    >>> backend = get_backend('auto')  # Uses GPU if available, else CPU
    """
    if backend == "auto":
        # Check environment variable first
        env_backend = os.environ.get("MUSEVAL_BACKEND", None)
        if env_backend is not None:
            backend = env_backend
        else:
            # Auto-detect: try GPU first, fall back to CPU
            backend = _detect_best_backend()

    backend = backend.lower()

    if backend == "numpy":
        from .numpy_backend import NumpyBackend

        return NumpyBackend()
    elif backend == "cupy":
        try:
            from .cupy_backend import CupyBackend

            return CupyBackend()
        except ImportError as e:
            warnings.warn(
                f"CuPy not available ({e}). Install with: "
                "pip install museval[gpu]. Falling back to NumPy.",
                UserWarning,
            )
            from .numpy_backend import NumpyBackend

            return NumpyBackend()
        except RuntimeError as e:
            warnings.warn(
                f"CUDA not available ({e}). Falling back to NumPy.", UserWarning
            )
            from .numpy_backend import NumpyBackend

            return NumpyBackend()
    else:
        raise ValueError(
            f"Unknown backend: '{backend}'. Valid options are: 'numpy', 'cupy', 'auto'"
        )


__all__ = ["get_backend"]
