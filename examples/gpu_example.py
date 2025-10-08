"""Example demonstrating GPU acceleration for BSS evaluation.

This script shows how to use the GPU backend for faster evaluation
of source separation metrics.
"""
import numpy as np
import museval.metrics as metrics
import time


def check_gpu_available():
    """Check if GPU is available."""
    try:
        import cupy as cp
        return cp.cuda.is_available()
    except (ImportError, RuntimeError):
        return False


def generate_test_data(nsrc=4, duration_sec=10, sample_rate=44100, nchan=2):
    """Generate random test data for evaluation."""
    nsampl = duration_sec * sample_rate
    
    # Generate random sources with some structure
    np.random.seed(42)
    reference = np.random.randn(nsrc, nsampl, nchan) * 0.5
    
    # Generate estimates that are somewhat correlated with reference
    estimated = np.random.randn(nsrc, nsampl, nchan) * 0.5
    estimated += reference * 0.3  # Add correlation
    
    return reference, estimated


def evaluate_with_timing(reference, estimated, backend='numpy', description=''):
    """Evaluate and time the computation."""
    print(f"\n{description}")
    print("-" * 60)
    
    start = time.time()
    sdr, isr, sir, sar, perm = metrics.bss_eval(
        reference, estimated,
        window=np.inf,
        hop=np.inf,
        compute_permutation=False,
        backend=backend
    )
    elapsed = time.time() - start
    
    print(f"Time: {elapsed:.3f} seconds")
    print(f"SDR: {sdr.mean():.2f} dB")
    print(f"ISR: {isr.mean():.2f} dB")
    print(f"SIR: {sir.mean():.2f} dB")
    print(f"SAR: {sar.mean():.2f} dB")
    
    return elapsed


def main():
    """Run the example."""
    print("=" * 60)
    print("GPU Acceleration Example for museval")
    print("=" * 60)
    
    has_gpu = check_gpu_available()
    
    if has_gpu:
        print("\n✓ GPU detected and available")
    else:
        print("\n⚠️  GPU not available")
        print("Install CuPy to enable GPU support:")
        print("  pip install museval[gpu]")
        print("\nContinuing with CPU-only demonstration...")
    
    # Generate test data
    print("\nGenerating test data (4 sources, 10 seconds, stereo)...")
    reference, estimated = generate_test_data(
        nsrc=4, duration_sec=10, sample_rate=44100, nchan=2
    )
    
    # Evaluate with CPU
    cpu_time = evaluate_with_timing(
        reference, estimated,
        backend='numpy',
        description='Evaluating with CPU (NumPy backend)'
    )
    
    # Evaluate with GPU if available
    if has_gpu:
        # Warmup run
        print("\nWarming up GPU...")
        _ = metrics.bss_eval(
            reference[:, :1000, :], estimated[:, :1000, :],
            backend='cupy'
        )
        
        # Actual evaluation
        gpu_time = evaluate_with_timing(
            reference, estimated,
            backend='cupy',
            description='Evaluating with GPU (CuPy backend)'
        )
        
        # Show speedup
        speedup = cpu_time / gpu_time
        print("\n" + "=" * 60)
        print(f"Speedup: {speedup:.2f}x")
        if speedup > 1:
            print("✓ GPU is faster!")
        else:
            print("⚠️  CPU is faster for this workload")
            print("   Try larger inputs for better GPU utilization")
        print("=" * 60)
    
    # Demonstrate environment variable usage
    print("\n" + "=" * 60)
    print("Alternative: Using environment variable")
    print("=" * 60)
    print("\nYou can set the backend globally using:")
    print("  export MUSEVAL_BACKEND=cupy")
    print("\nThen in your code:")
    print("  sdr, isr, sir, sar, perm = metrics.bss_eval(")
    print("      reference, estimated")
    print("  )")
    print("\nThe backend will automatically use GPU if available!")
    
    # Demonstrate wrapper functions
    print("\n" + "=" * 60)
    print("Using wrapper functions")
    print("=" * 60)
    print("\nAll wrapper functions support the backend parameter:")
    print("  - bss_eval_sources(reference, estimated, backend='cupy')")
    print("  - bss_eval_images(reference, estimated, backend='cupy')")
    print("  - bss_eval_sources_framewise(..., backend='cupy')")
    print("  - bss_eval_images_framewise(..., backend='cupy')")


if __name__ == '__main__':
    main()

