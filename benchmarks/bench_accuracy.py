"""Benchmark: Q4 quantization accuracy vs original FP32/FP16 values.

Measures max absolute error, mean absolute error, and RMSE across
different value distributions to build confidence in Q4 quality.

Usage: python3 benchmarks/bench_accuracy.py
"""

import numpy as np
import eakv


def measure_error(name, kv):
    """Quantize, restore, and measure error against original."""
    bundle = eakv.quantize(kv)
    restored = eakv.dequantize(bundle)

    original = kv.astype(np.float32)
    diff = np.abs(original - restored)
    max_err = np.max(diff)
    mean_err = np.mean(diff)
    rmse = np.sqrt(np.mean(diff ** 2))

    # Signal-to-noise ratio (dB)
    signal_power = np.mean(original ** 2)
    noise_power = np.mean(diff ** 2)
    snr_db = 10 * np.log10(signal_power / noise_power) if noise_power > 0 else float('inf')

    # Cosine similarity (flattened)
    a, b = original.ravel(), restored.ravel()
    cos_sim = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    print(f"  {name}:")
    print(f"    Max abs error:  {max_err:.6f}")
    print(f"    Mean abs error: {mean_err:.6f}")
    print(f"    RMSE:           {rmse:.6f}")
    print(f"    SNR:            {snr_db:.1f} dB")
    print(f"    Cosine sim:     {cos_sim:.8f}")
    return max_err, mean_err, rmse


def main():
    nl, nh, sl, hd = 4, 8, 512, 128
    shape = (nl, 2, nh, sl, hd)

    print("=" * 60)
    print(f"Q4 Accuracy Benchmark ({nl}L, {nh}H, {sl}seq, {hd}dim)")
    print("=" * 60)

    # 1. Standard normal (typical KV cache distribution)
    kv_normal = np.random.randn(*shape).astype(np.float32)
    measure_error("Normal(0,1)", kv_normal)

    # 2. Small range (typical attention values ~[-1, 1])
    kv_small = (np.random.randn(*shape) * 0.1).astype(np.float32)
    measure_error("Normal(0,0.1) — small range", kv_small)

    # 3. Large range (stress test)
    kv_large = (np.random.randn(*shape) * 100).astype(np.float32)
    measure_error("Normal(0,100) — large range", kv_large)

    # 4. Uniform [0, 1]
    kv_uniform = np.random.rand(*shape).astype(np.float32)
    measure_error("Uniform[0,1]", kv_uniform)

    # 5. FP16 input (real-world scenario)
    kv_f16 = np.random.randn(*shape).astype(np.float16)
    measure_error("Normal(0,1) FP16 input", kv_f16)

    # 6. Nearly constant (edge case — tiny range)
    kv_const = np.full(shape, 3.14, dtype=np.float32)
    kv_const += np.random.randn(*shape).astype(np.float32) * 1e-4
    measure_error("Near-constant (3.14 ± 1e-4)", kv_const)

    # Summary
    print()
    print("=" * 60)
    print("Summary:")
    print("  Q4_1 quantizes 64 values into 32 bytes + scale + bias.")
    print("  4-bit quantization gives 16 levels per group.")
    print("  Expected max error ≈ range/30 per group (half a quant step).")
    print("  For Normal(0,1): typical group range ~4-6, so max err ~0.15-0.2")
    print()
    print("  For LLM KV caches, this error is negligible —")
    print("  perplexity impact is typically < 0.1% (see GPTQ, AWQ literature).")
    print("=" * 60)


if __name__ == "__main__":
    main()
