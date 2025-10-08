"""Tests for backend abstraction layer."""
import numpy as np
import pytest
from museval.backends import get_backend


def _cupy_available():
    """Check if CuPy is available."""
    try:
        import cupy as cp
        return cp.cuda.is_available()
    except (ImportError, RuntimeError):
        return False


def test_backend_selection_numpy():
    """Test that NumPy backend can be selected."""
    backend = get_backend('numpy')
    assert backend.name == 'numpy'
    assert backend.array_type == np.ndarray


def test_backend_selection_auto_defaults_to_numpy():
    """Test that auto mode defaults to NumPy."""
    backend = get_backend('auto')
    assert backend.name == 'numpy'


def test_backend_selection_invalid():
    """Test that invalid backend raises ValueError."""
    with pytest.raises(ValueError, match="Unknown backend"):
        get_backend('invalid_backend')


def test_numpy_backend_basic_operations():
    """Test basic operations with NumPy backend."""
    backend = get_backend('numpy')
    
    # Array creation
    arr = backend.zeros((3, 3))
    assert arr.shape == (3, 3)
    assert np.all(arr == 0)
    
    # Math operations
    arr2 = backend.array([1, 2, 3])
    assert backend.sum(arr2) == 6
    assert backend.mean(arr2) == 2.0
    
    # FFT
    fft_result = backend.fft(arr2)
    assert fft_result.shape == arr2.shape


def test_numpy_backend_asnumpy():
    """Test that NumPy backend asnumpy returns numpy array."""
    backend = get_backend('numpy')
    arr = backend.array([1, 2, 3])
    result = backend.asnumpy(arr)
    assert isinstance(result, np.ndarray)


def test_numpy_backend_device_info():
    """Test device info for NumPy backend."""
    backend = get_backend('numpy')
    device_info = backend.get_device()
    assert device_info['type'] == 'cpu'


@pytest.mark.skipif(
    not _cupy_available(),
    reason="CuPy not available"
)
def test_backend_selection_cupy():
    """Test that CuPy backend can be selected if available."""
    backend = get_backend('cupy')
    assert backend.name == 'cupy'


@pytest.mark.skipif(
    not _cupy_available(),
    reason="CuPy not available"
)
def test_cupy_backend_basic_operations():
    """Test basic operations with CuPy backend."""
    backend = get_backend('cupy')
    
    # Array creation
    arr = backend.zeros((3, 3))
    assert arr.shape == (3, 3)
    
    # Math operations
    arr2 = backend.array([1, 2, 3])
    assert float(backend.sum(arr2)) == 6
    
    # FFT
    fft_result = backend.fft(arr2)
    assert fft_result.shape == arr2.shape


@pytest.mark.skipif(
    not _cupy_available(),
    reason="CuPy not available"
)
def test_cupy_backend_asnumpy():
    """Test that CuPy backend asnumpy returns numpy array."""
    backend = get_backend('cupy')
    arr = backend.array([1, 2, 3])
    result = backend.asnumpy(arr)
    assert isinstance(result, np.ndarray)
    np.testing.assert_array_equal(result, np.array([1, 2, 3]))


@pytest.mark.skipif(
    not _cupy_available(),
    reason="CuPy not available"
)
def test_cupy_backend_device_info():
    """Test device info for CuPy backend."""
    backend = get_backend('cupy')
    device_info = backend.get_device()
    assert device_info['type'] == 'cuda'
    assert 'id' in device_info


def test_backend_fallback_warning():
    """Test that attempting to use CuPy without installation issues warning."""
    # Only test if CuPy is NOT available
    if _cupy_available():
        pytest.skip("CuPy is available, cannot test fallback")
    
    with pytest.warns(UserWarning, match="CuPy not available"):
        backend = get_backend('cupy')
        assert backend.name == 'numpy'

