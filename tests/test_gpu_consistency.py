"""Tests for CPU/GPU consistency of BSS evaluation metrics.

These tests verify that the GPU (CuPy) backend produces numerically
consistent results with the CPU (NumPy) backend.
"""
import numpy as np
import pytest
import museval.metrics as metrics


def _cupy_available():
    """Check if CuPy is available."""
    try:
        import cupy as cp
        return cp.cuda.is_available()
    except (ImportError, RuntimeError):
        return False


HAS_GPU = _cupy_available()
GPU_SKIP_REASON = "GPU not available or CuPy not installed"


@pytest.fixture
def random_sources():
    """Generate random reference and estimated sources."""
    def _generate(nsrc=2, nsampl=44100, nchan=2, seed=42):
        np.random.seed(seed)
        reference = np.random.randn(nsrc, nsampl, nchan) * 0.5
        estimated = np.random.randn(nsrc, nsampl, nchan) * 0.5
        # Make them somewhat correlated
        estimated += reference * 0.3
        return reference, estimated
    return _generate


@pytest.mark.skipif(not HAS_GPU, reason=GPU_SKIP_REASON)
@pytest.mark.parametrize("nsrc,nsampl,nchan", [
    (2, 8000, 1),       # Stereo sources, mono
    (2, 8000, 2),       # Stereo sources, stereo
    (4, 8000, 2),       # Many sources
    (1, 8000, 1),       # Single source
])
def test_bss_eval_cpu_gpu_consistency_wholefile(random_sources, nsrc, nsampl, nchan):
    """Test that CPU and GPU produce identical results for whole file evaluation."""
    reference, estimated = random_sources(nsrc, nsampl, nchan)
    
    # Compute on CPU
    sdr_cpu, isr_cpu, sir_cpu, sar_cpu, perm_cpu = metrics.bss_eval(
        reference, estimated,
        window=np.inf,
        hop=np.inf,
        compute_permutation=False,
        backend='numpy'
    )
    
    # Compute on GPU
    sdr_gpu, isr_gpu, sir_gpu, sar_gpu, perm_gpu = metrics.bss_eval(
        reference, estimated,
        window=np.inf,
        hop=np.inf,
        compute_permutation=False,
        backend='cupy'
    )
    
    # Compare results with tolerance
    np.testing.assert_allclose(sdr_cpu, sdr_gpu, rtol=1e-4, atol=1e-5,
                               err_msg="SDR values differ between CPU and GPU")
    np.testing.assert_allclose(isr_cpu, isr_gpu, rtol=1e-4, atol=1e-5,
                               err_msg="ISR values differ between CPU and GPU")
    np.testing.assert_allclose(sir_cpu, sir_gpu, rtol=1e-4, atol=1e-5,
                               err_msg="SIR values differ between CPU and GPU")
    np.testing.assert_allclose(sar_cpu, sar_gpu, rtol=1e-4, atol=1e-5,
                               err_msg="SAR values differ between CPU and GPU")
    np.testing.assert_array_equal(perm_cpu, perm_gpu,
                                  err_msg="Permutations differ between CPU and GPU")


@pytest.mark.skipif(not HAS_GPU, reason=GPU_SKIP_REASON)
@pytest.mark.parametrize("window,hop", [
    (4000, 2000),           # Overlapping windows
    (8000, 4000),           # Larger windows
    (2000, 2000),           # Non-overlapping
])
def test_bss_eval_cpu_gpu_consistency_framewise(random_sources, window, hop):
    """Test CPU/GPU consistency for framewise evaluation."""
    reference, estimated = random_sources(nsrc=2, nsampl=10000, nchan=2)
    
    # Compute on CPU
    sdr_cpu, isr_cpu, sir_cpu, sar_cpu, perm_cpu = metrics.bss_eval(
        reference, estimated,
        window=window,
        hop=hop,
        compute_permutation=False,
        framewise_filters=False,
        backend='numpy'
    )
    
    # Compute on GPU
    sdr_gpu, isr_gpu, sir_gpu, sar_gpu, perm_gpu = metrics.bss_eval(
        reference, estimated,
        window=window,
        hop=hop,
        compute_permutation=False,
        framewise_filters=False,
        backend='cupy'
    )
    
    # Compare results
    np.testing.assert_allclose(sdr_cpu, sdr_gpu, rtol=1e-4, atol=1e-5)
    np.testing.assert_allclose(isr_cpu, isr_gpu, rtol=1e-4, atol=1e-5)
    np.testing.assert_allclose(sir_cpu, sir_gpu, rtol=1e-4, atol=1e-5)
    np.testing.assert_allclose(sar_cpu, sar_gpu, rtol=1e-4, atol=1e-5)
    np.testing.assert_array_equal(perm_cpu, perm_gpu)


@pytest.mark.skipif(not HAS_GPU, reason=GPU_SKIP_REASON)
def test_bss_eval_cpu_gpu_consistency_with_permutation(random_sources):
    """Test CPU/GPU consistency when computing permutations."""
    reference, estimated = random_sources(nsrc=3, nsampl=6000, nchan=2)
    
    # Compute on CPU
    sdr_cpu, isr_cpu, sir_cpu, sar_cpu, perm_cpu = metrics.bss_eval(
        reference, estimated,
        window=np.inf,
        hop=np.inf,
        compute_permutation=True,
        backend='numpy'
    )
    
    # Compute on GPU
    sdr_gpu, isr_gpu, sir_gpu, sar_gpu, perm_gpu = metrics.bss_eval(
        reference, estimated,
        window=np.inf,
        hop=np.inf,
        compute_permutation=True,
        backend='cupy'
    )
    
    # Compare results
    np.testing.assert_allclose(sdr_cpu, sdr_gpu, rtol=1e-4, atol=1e-5)
    np.testing.assert_allclose(isr_cpu, isr_gpu, rtol=1e-4, atol=1e-5)
    np.testing.assert_allclose(sir_cpu, sir_gpu, rtol=1e-4, atol=1e-5)
    np.testing.assert_allclose(sar_cpu, sar_gpu, rtol=1e-4, atol=1e-5)
    np.testing.assert_array_equal(perm_cpu, perm_gpu)


@pytest.mark.skipif(not HAS_GPU, reason=GPU_SKIP_REASON)
def test_bss_eval_sources_cpu_gpu_consistency(random_sources):
    """Test CPU/GPU consistency for bss_eval_sources wrapper."""
    reference, estimated = random_sources(nsrc=2, nsampl=8000, nchan=2)
    
    # Compute on CPU
    sdr_cpu, sir_cpu, sar_cpu, perm_cpu = metrics.bss_eval_sources(
        reference, estimated,
        compute_permutation=False,
        backend='numpy'
    )
    
    # Compute on GPU
    sdr_gpu, sir_gpu, sar_gpu, perm_gpu = metrics.bss_eval_sources(
        reference, estimated,
        compute_permutation=False,
        backend='cupy'
    )
    
    # Compare results
    np.testing.assert_allclose(sdr_cpu, sdr_gpu, rtol=1e-4, atol=1e-5)
    np.testing.assert_allclose(sir_cpu, sir_gpu, rtol=1e-4, atol=1e-5)
    np.testing.assert_allclose(sar_cpu, sar_gpu, rtol=1e-4, atol=1e-5)
    np.testing.assert_array_equal(perm_cpu, perm_gpu)


@pytest.mark.skipif(not HAS_GPU, reason=GPU_SKIP_REASON)
def test_bss_eval_images_cpu_gpu_consistency(random_sources):
    """Test CPU/GPU consistency for bss_eval_images wrapper."""
    reference, estimated = random_sources(nsrc=2, nsampl=8000, nchan=2)
    
    # Compute on CPU
    sdr_cpu, isr_cpu, sir_cpu, sar_cpu, perm_cpu = metrics.bss_eval_images(
        reference, estimated,
        compute_permutation=False,
        backend='numpy'
    )
    
    # Compute on GPU
    sdr_gpu, isr_gpu, sir_gpu, sar_gpu, perm_gpu = metrics.bss_eval_images(
        reference, estimated,
        compute_permutation=False,
        backend='cupy'
    )
    
    # Compare results
    np.testing.assert_allclose(sdr_cpu, sdr_gpu, rtol=1e-4, atol=1e-5)
    np.testing.assert_allclose(isr_cpu, isr_gpu, rtol=1e-4, atol=1e-5)
    np.testing.assert_allclose(sir_cpu, sir_gpu, rtol=1e-4, atol=1e-5)
    np.testing.assert_allclose(sar_cpu, sar_gpu, rtol=1e-4, atol=1e-5)
    np.testing.assert_array_equal(perm_cpu, perm_gpu)


@pytest.mark.skipif(not HAS_GPU, reason=GPU_SKIP_REASON)
def test_bss_eval_images_framewise_cpu_gpu_consistency(random_sources):
    """Test CPU/GPU consistency for bss_eval_images_framewise wrapper."""
    reference, estimated = random_sources(nsrc=2, nsampl=12000, nchan=2)
    
    # Compute on CPU
    sdr_cpu, isr_cpu, sir_cpu, sar_cpu, perm_cpu = metrics.bss_eval_images_framewise(
        reference, estimated,
        window=6000,
        hop=3000,
        compute_permutation=False,
        backend='numpy'
    )
    
    # Compute on GPU
    sdr_gpu, isr_gpu, sir_gpu, sar_gpu, perm_gpu = metrics.bss_eval_images_framewise(
        reference, estimated,
        window=6000,
        hop=3000,
        compute_permutation=False,
        backend='cupy'
    )
    
    # Compare results
    np.testing.assert_allclose(sdr_cpu, sdr_gpu, rtol=1e-4, atol=1e-5)
    np.testing.assert_allclose(isr_cpu, isr_gpu, rtol=1e-4, atol=1e-5)
    np.testing.assert_allclose(sir_cpu, sir_gpu, rtol=1e-4, atol=1e-5)
    np.testing.assert_allclose(sar_cpu, sar_gpu, rtol=1e-4, atol=1e-5)
    np.testing.assert_array_equal(perm_cpu, perm_gpu)


@pytest.mark.skipif(not HAS_GPU, reason=GPU_SKIP_REASON)
@pytest.mark.parametrize("filters_len", [256, 512, 1024])
def test_bss_eval_cpu_gpu_consistency_different_filter_lengths(random_sources, filters_len):
    """Test CPU/GPU consistency with different filter lengths."""
    reference, estimated = random_sources(nsrc=2, nsampl=8000, nchan=2)
    
    # Compute on CPU
    sdr_cpu, isr_cpu, sir_cpu, sar_cpu, perm_cpu = metrics.bss_eval(
        reference, estimated,
        window=np.inf,
        hop=np.inf,
        filters_len=filters_len,
        compute_permutation=False,
        backend='numpy'
    )
    
    # Compute on GPU
    sdr_gpu, isr_gpu, sir_gpu, sar_gpu, perm_gpu = metrics.bss_eval(
        reference, estimated,
        window=np.inf,
        hop=np.inf,
        filters_len=filters_len,
        compute_permutation=False,
        backend='cupy'
    )
    
    # Compare results
    np.testing.assert_allclose(sdr_cpu, sdr_gpu, rtol=1e-4, atol=1e-5)
    np.testing.assert_allclose(isr_cpu, isr_gpu, rtol=1e-4, atol=1e-5)
    np.testing.assert_allclose(sir_cpu, sir_gpu, rtol=1e-4, atol=1e-5)
    np.testing.assert_allclose(sar_cpu, sar_gpu, rtol=1e-4, atol=1e-5)


@pytest.mark.skipif(not HAS_GPU, reason=GPU_SKIP_REASON)
def test_empty_input_cpu_gpu_consistency():
    """Test that empty inputs produce consistent results on CPU and GPU."""
    inputs = [np.array([]), np.array([])]
    
    with pytest.warns(UserWarning):
        output_cpu = metrics.bss_eval(*inputs, backend='numpy')
    
    with pytest.warns(UserWarning):
        output_gpu = metrics.bss_eval(*inputs, backend='cupy')
    
    # Both should return empty arrays
    for cpu_arr, gpu_arr in zip(output_cpu, output_gpu):
        assert cpu_arr.size == 0
        assert gpu_arr.size == 0


@pytest.mark.skipif(not HAS_GPU, reason=GPU_SKIP_REASON)
def test_silent_input_cpu_gpu_consistency(random_sources):
    """Test that silent inputs raise the same error on CPU and GPU."""
    reference, _ = random_sources(nsrc=2, nsampl=8000, nchan=2)
    estimated = np.zeros(reference.shape)
    
    # Both should raise ValueError
    with pytest.raises(ValueError):
        metrics.bss_eval(reference, estimated, backend='numpy')
    
    with pytest.raises(ValueError):
        metrics.bss_eval(reference, estimated, backend='cupy')


def test_cpu_backend_works_without_gpu():
    """Test that CPU backend always works regardless of GPU availability."""
    np.random.seed(42)
    reference = np.random.randn(2, 4000, 2)
    estimated = np.random.randn(2, 4000, 2)
    
    # This should always work
    sdr, isr, sir, sar, perm = metrics.bss_eval(
        reference, estimated,
        window=np.inf,
        hop=np.inf,
        backend='numpy'
    )
    
    # Check that we got valid results
    assert sdr.shape[0] == 2  # 2 sources
    assert not np.any(np.isnan(sdr))


@pytest.mark.skipif(not HAS_GPU, reason=GPU_SKIP_REASON)
def test_bss_eval_cpu_gpu_consistency_bsseval_sources_version(random_sources):
    """Test CPU/GPU consistency with bsseval_sources_version=True."""
    reference, estimated = random_sources(nsrc=2, nsampl=8000, nchan=2)
    
    # Compute on CPU
    sdr_cpu, isr_cpu, sir_cpu, sar_cpu, perm_cpu = metrics.bss_eval(
        reference, estimated,
        window=np.inf,
        hop=np.inf,
        compute_permutation=False,
        bsseval_sources_version=True,
        backend='numpy'
    )
    
    # Compute on GPU
    sdr_gpu, isr_gpu, sir_gpu, sar_gpu, perm_gpu = metrics.bss_eval(
        reference, estimated,
        window=np.inf,
        hop=np.inf,
        compute_permutation=False,
        bsseval_sources_version=True,
        backend='cupy'
    )
    
    # Compare results
    np.testing.assert_allclose(sdr_cpu, sdr_gpu, rtol=1e-4, atol=1e-5)
    # ISR should be NaN for bsseval_sources_version
    assert np.all(np.isnan(isr_cpu))
    assert np.all(np.isnan(isr_gpu))
    np.testing.assert_allclose(sir_cpu, sir_gpu, rtol=1e-4, atol=1e-5)
    np.testing.assert_allclose(sar_cpu, sar_gpu, rtol=1e-4, atol=1e-5)
    np.testing.assert_array_equal(perm_cpu, perm_gpu)

