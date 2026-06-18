"""Benchmark script to compare CPU vs GPU performance.

Run this script to measure the speedup achieved by GPU acceleration
for various input sizes and configurations.
"""
import numpy as np
import time
import museval.metrics as metrics


def _cupy_available():
    """Check if CuPy is available."""
    try:
        import cupy as cp
        return cp.cuda.is_available()
    except (ImportError, RuntimeError):
        return False


def benchmark_bss_eval(nsrc, nsampl, nchan, window, hop, backend='numpy', warmup=True):
    """Benchmark bss_eval with given parameters.
    
    Parameters
    ----------
    nsrc : int
        Number of sources
    nsampl : int
        Number of samples
    nchan : int
        Number of channels
    window : int or float
        Window size
    hop : int or float
        Hop size
    backend : str
        Backend to use ('numpy' or 'cupy')
    warmup : bool
        Whether to do a warmup run (important for GPU)
        
    Returns
    -------
    elapsed_time : float
        Time in seconds
    """
    # Generate test data
    np.random.seed(42)
    reference = np.random.randn(nsrc, nsampl, nchan) * 0.5
    estimated = np.random.randn(nsrc, nsampl, nchan) * 0.5
    estimated += reference * 0.3  # Add some correlation
    
    # Warmup run for GPU
    if warmup and backend == 'cupy':
        try:
            _ = metrics.bss_eval(
                reference, estimated,
                window=window, hop=hop,
                compute_permutation=False,
                backend=backend
            )
        except Exception as e:
            print(f"Warmup failed: {e}")
            return None
    
    # Benchmark run
    start = time.time()
    try:
        sdr, isr, sir, sar, perm = metrics.bss_eval(
            reference, estimated,
            window=window, hop=hop,
            compute_permutation=False,
            backend=backend
        )
        elapsed = time.time() - start
        return elapsed
    except Exception as e:
        print(f"Benchmark failed for {backend}: {e}")
        return None


def format_time(seconds):
    """Format time in seconds to readable string."""
    if seconds < 0.001:
        return f"{seconds * 1e6:.1f} µs"
    elif seconds < 1:
        return f"{seconds * 1e3:.1f} ms"
    else:
        return f"{seconds:.2f} s"


def main():
    """Run benchmarks and print results."""
    print("=" * 80)
    print("BSS Eval CPU vs GPU Benchmark")
    print("=" * 80)
    
    has_gpu = _cupy_available()
    if not has_gpu:
        print("\n⚠️  GPU not available. Only CPU benchmarks will be run.")
        print("Install CuPy to enable GPU benchmarks: pip install museval[gpu]\n")
    else:
        print("\n✓ GPU detected and available\n")
    
    # Test configurations
    configs = [
        # (nsrc, nsampl, nchan, window, hop, description)
        (2, 44100, 2, np.inf, np.inf, "Short stereo (1s, 2 sources)"),
        (2, 44100 * 10, 2, np.inf, np.inf, "Long stereo (10s, 2 sources)"),
        (4, 44100 * 10, 2, np.inf, np.inf, "Long stereo (10s, 4 sources)"),
        (2, 44100 * 30, 2, np.inf, np.inf, "Very long stereo (30s, 2 sources)"),
        (2, 44100 * 10, 2, 44100 * 2, 44100, "Framewise (10s, 2s windows)"),
        (4, 44100 * 10, 2, 44100 * 2, 44100, "Framewise (10s, 4 sources)"),
    ]
    
    results = []
    
    for nsrc, nsampl, nchan, window, hop, description in configs:
        print(f"\nBenchmarking: {description}")
        print(f"  Config: {nsrc} sources, {nsampl} samples, {nchan} channels")
        
        # CPU benchmark
        cpu_time = benchmark_bss_eval(nsrc, nsampl, nchan, window, hop, 'numpy')
        if cpu_time is not None:
            print(f"  CPU time: {format_time(cpu_time)}")
        else:
            print(f"  CPU time: FAILED")
            continue
        
        # GPU benchmark
        if has_gpu:
            gpu_time = benchmark_bss_eval(nsrc, nsampl, nchan, window, hop, 'cupy')
            if gpu_time is not None:
                print(f"  GPU time: {format_time(gpu_time)}")
                speedup = cpu_time / gpu_time
                print(f"  Speedup: {speedup:.2f}x")
                
                if speedup > 1:
                    print(f"  ✓ GPU is faster")
                else:
                    print(f"  ⚠️  CPU is faster (GPU overhead dominates)")
                
                results.append({
                    'description': description,
                    'cpu_time': cpu_time,
                    'gpu_time': gpu_time,
                    'speedup': speedup
                })
            else:
                print(f"  GPU time: FAILED")
    
    # Summary
    if has_gpu and results:
        print("\n" + "=" * 80)
        print("Summary")
        print("=" * 80)
        print(f"\n{'Description':<45} {'CPU':<12} {'GPU':<12} {'Speedup':<10}")
        print("-" * 80)
        
        for r in results:
            print(f"{r['description']:<45} "
                  f"{format_time(r['cpu_time']):<12} "
                  f"{format_time(r['gpu_time']):<12} "
                  f"{r['speedup']:.2f}x")
        
        avg_speedup = np.mean([r['speedup'] for r in results])
        print("-" * 80)
        print(f"Average speedup: {avg_speedup:.2f}x")
        
        print("\nRecommendations:")
        if avg_speedup > 2:
            print("  ✓ GPU acceleration provides significant speedup for these workloads")
        elif avg_speedup > 1.2:
            print("  ✓ GPU acceleration provides moderate speedup")
        else:
            print("  ⚠️  GPU overhead may dominate for these workloads")
            print("     Consider using GPU for larger inputs or batch processing")


if __name__ == '__main__':
    main()

