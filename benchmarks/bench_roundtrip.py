"""Benchmark: quantize/restore times for various KV cache sizes.

Usage: python3 benchmarks/bench_roundtrip.py
"""

import time
import tempfile
import numpy as np
import eakv


def bench(name, fn, warmup=1, runs=5):
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    avg = np.mean(times)
    std = np.std(times)
    print(f"  {name}: {avg*1000:.1f}ms +/- {std*1000:.1f}ms")
    return avg


def main():
    configs = [
        # ("Small (1B-like)", 12, 8, 512, 64),
        ("Medium (7B-like)", 32, 8, 2048, 128),
        # ("Large (8B-like)", 32, 32, 4096, 128),
    ]

    for name, nl, nh, sl, hd in configs:
        print(f"\n{'='*60}")
        print(f"{name}: {nl} layers, {nh} heads, {sl} seq, {hd} dim")

        kv = np.random.randn(nl, 2, nh, sl, hd).astype(np.float32)
        size_mb = kv.nbytes / 1024 / 1024
        print(f"  Original size: {size_mb:.1f} MB")

        t_q = bench("Quantize", lambda: eakv.quantize(kv))
        bundle = eakv.quantize(kv)
        comp_mb = bundle.compressed_size / 1024 / 1024
        print(f"  Compressed size: {comp_mb:.1f} MB ({bundle.compression_ratio:.1%})")

        t_r = bench("Full restore", lambda: eakv.dequantize(bundle))

        if sl >= 256:
            bench("Partial restore (last 256 tokens)", lambda: eakv.restore(bundle, tokens=-256))

        with tempfile.NamedTemporaryFile(suffix=".eakv") as f:
            bench("Save to disk", lambda: eakv.save(bundle, f.name))
            bench("Load from disk", lambda: eakv.load(f.name))

            eakv.save(bundle, f.name)
            bench("Mmap open + restore", lambda: _mmap_restore(f.name))

        print(f"  Quantize: {t_q*1000:.0f}ms, Restore: {t_r*1000:.0f}ms")


def _mmap_restore(path):
    with eakv.open_mmap(path) as b:
        return eakv.dequantize(b)


if __name__ == "__main__":
    main()
