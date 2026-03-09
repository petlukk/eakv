"""Benchmark: fused attention vs restore-then-compute.

The fair comparison includes dequantization time in the baseline,
since the whole point of fusion is to eliminate the restore step.

Usage: python3 benchmarks/bench_attention.py
"""

import sys
import time
import numpy as np

sys.path.insert(0, "src")
from eakv._quantize import quantize
from eakv._restore import dequantize, restore
from eakv._attention import (
    attention_scores, attention_output,
    attention_scores_multi, attention_output_multi,
)
from eakv._dispatch import q4_dequantize_dispatch


def _softmax(x):
    e = np.exp(x - np.max(x))
    return e / e.sum()


def bench(name, fn, warmup=3, runs=10):
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    avg = np.mean(times) * 1e6
    std = np.std(times) * 1e6
    print(f"  {name:<45s} {avg:>8.0f} us +/- {std:>6.0f}")
    return avg


def main():
    configs = [
        ("7B-like", 1, 8, 2048, 128),
        ("7B-like (long)", 1, 8, 8192, 128),
    ]

    for name, nl, nh, sl, hd in configs:
        print(f"\n{'='*65}")
        print(f"{name}: {nh} heads, {sl} seq, {hd} dim (single layer, single head)")

        kv = np.random.default_rng(42).standard_normal(
            (nl, 2, nh, sl, hd)
        ).astype(np.float32)
        bundle = quantize(kv)
        restored = dequantize(bundle)

        q_vec = np.random.default_rng(99).standard_normal(hd).astype(np.float32)

        # Per-head group count and slicing
        groups_per_head = sl * 2
        g_start = 0
        g_end = groups_per_head
        w_start = g_start * 32
        w_end = g_end * 32

        head_weights = bundle.weights[0, 0, w_start:w_end].copy()
        head_scales = bundle.scales[0, 0, g_start:g_end].copy()
        head_biases = bundle.biases[0, 0, g_start:g_end].copy()

        # --- K scores ---
        print("\n  K-score (query @ K^T / sqrt(d)):")

        # Baseline 1: numpy matmul on pre-dequantized data (unrealistic best case)
        k_head = restored[0, 0, 0].copy()
        bench("NumPy matmul (data already in f32)",
              lambda: k_head @ q_vec / np.sqrt(128.0))

        # Baseline 2: dequantize head + numpy matmul (fair comparison)
        def restore_then_score():
            flat = np.empty(groups_per_head * 64, dtype=np.float32)
            q4_dequantize_dispatch(head_weights, head_scales, head_biases,
                                   flat, groups_per_head)
            k = flat.reshape(sl, hd)
            return k @ q_vec / np.sqrt(128.0)

        t_base = bench("Restore head + numpy matmul",
                       restore_then_score)

        # Fused
        t_fused = bench("Fused kernel (no restore needed)",
                        lambda: attention_scores(bundle, q_vec, layer=0, head=0))

        print(f"  Speedup vs restore+matmul: {t_base / t_fused:.2f}x")

        # --- V weighted sum ---
        print("\n  V-sum (softmax @ V):")
        raw = np.random.default_rng(77).standard_normal(sl).astype(np.float32)
        w = _softmax(raw)

        v_head = restored[0, 1, 0].copy()
        bench("NumPy matmul (data already in f32)",
              lambda: w @ v_head)

        head_v_weights = bundle.weights[0, 1, w_start:w_end].copy()
        head_v_scales = bundle.scales[0, 1, g_start:g_end].copy()
        head_v_biases = bundle.biases[0, 1, g_start:g_end].copy()

        def restore_then_vsum():
            flat = np.empty(groups_per_head * 64, dtype=np.float32)
            q4_dequantize_dispatch(head_v_weights, head_v_scales, head_v_biases,
                                   flat, groups_per_head)
            v = flat.reshape(sl, hd)
            return w @ v

        t_base = bench("Restore head + numpy matmul",
                       restore_then_vsum)

        t_fused = bench("Fused kernel (no restore needed)",
                        lambda: attention_output(bundle, w, layer=0, head=0))

        print(f"  Speedup vs restore+matmul: {t_base / t_fused:.2f}x")

        # --- Multi-head comparison ---
        print(f"\n  Multi-head ({nh} heads, K-score + V-sum combined):")

        q_vecs = np.random.default_rng(99).standard_normal((nh, hd)).astype(np.float32)
        all_w = np.stack([_softmax(np.random.default_rng(77 + h).standard_normal(sl).astype(np.float32))
                          for h in range(nh)])

        def per_head_loop():
            scores = np.empty((nh, sl), dtype=np.float32)
            outs = np.empty((nh, hd), dtype=np.float32)
            for h in range(nh):
                scores[h] = attention_scores(bundle, q_vecs[h], layer=0, head=h)
                outs[h] = attention_output(bundle, all_w[h], layer=0, head=h)
            return scores, outs

        t_loop = bench(f"Per-head Python loop ({nh} heads)",
                       per_head_loop)

        def multi_call():
            scores = attention_scores_multi(bundle, q_vecs, layer=0, n_heads=nh)
            outs = attention_output_multi(bundle, all_w, layer=0, n_heads=nh)
            return scores, outs

        t_multi = bench(f"Multi-head kernel ({nh} heads)",
                        multi_call)

        saved = t_loop - t_multi
        print(f"  Saved: {saved:.0f} us ({saved/t_loop*100:.1f}%), "
              f"~{saved/nh:.0f} us/head ctypes overhead")


if __name__ == "__main__":
    main()
