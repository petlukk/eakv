# Fused Attention-Dequantization Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Two Ea SIMD kernels that compute attention scores and weighted V sums directly from packed Q4 KV cache, eliminating the intermediate f32 restore step.

**Architecture:** Two fused kernels (K-score, V-sum) using AVX-512. Each loads packed Q4 bytes, dequantizes inline via widen+FMA, and immediately computes the attention operation. Dequantized values never touch memory. Python wrappers expose `attention_scores()` and `attention_output()` on the Q4Bundle.

**Tech Stack:** Ea kernel language (compiled via `ea` to .so), Python/NumPy for API, ctypes for FFI, pytest for testing.

**Design doc:** `docs/plans/2026-03-09-fused-attention-design.md`

---

### Task 1: K-Score Kernel

Write the Ea kernel that computes `scores[t] = dot(q_vec, dequant(K[t])) / sqrt(128)` for each sequence position t.

**Files:**
- Create: `kernels/fused_k_score.ea`

**Context:**
- Each sequence position's 128 dimensions span 2 consecutive groups (group `t*2` and `t*2+1`)
- Each group = 64 values packed into 32 bytes using split packing (lo nibble = values 0-31, hi nibble = values 32-63)
- The existing dequant pattern (from `kernels/dequantize_avx512.ea`) is: load u8x16, mask lo/hi nibbles, `widen_u8_f32x16`, `fma(widened, scale, bias)`
- Instead of storing the dequantized f32, we multiply by the corresponding Q vector chunk and `reduce_add` to accumulate a scalar dot product
- The `reduce_add` intrinsic is already supported in Ea (see `src/codegen/simd.rs:95`)
- Element-wise multiply uses `.*` operator in Ea
- `to_f32(integer)` converts scalar i32 to f32

**Step 1: Write the kernel**

```ea
// Fused K-score: dot(query, dequant(K)) per sequence position
//
// For each position t (head_dim=128, 2 groups per position):
//   Load packed K bytes, dequant inline, dot with query vector.
//   Eliminates intermediate f32 store — dequantized values stay in registers.

export func q4_fused_k_score_f32(
    q_vec:    *restrict f32,
    k_packed: *restrict u8,
    k_scales: *restrict f32,
    k_biases: *restrict f32,
    scores:   *mut f32,
    seq_len:  i32
) {
    let mask15: u8x16 = splat(15)
    let div16: u8x16 = splat(16)

    // Precompute 1/sqrt(128)
    let rsqrt_d: f32 = 1.0 / sqrt(to_f32(128))

    let mut t: i32 = 0
    while t < seq_len {
        let mut acc: f32 = 0.0

        // --- Group 0: dimensions 0-63 ---
        let g0: i32 = t * 2
        let vs0: f32x16 = splat(k_scales[g0])
        let vb0: f32x16 = splat(k_biases[g0])
        let wb0: i32 = g0 * 32
        let qb0: i32 = 0

        let p0a: u8x16 = load(k_packed, wb0)
        let lo0a: u8x16 = p0a .& mask15
        let hi0a: u8x16 = p0a ./ div16

        let p0b: u8x16 = load(k_packed, wb0 + 16)
        let lo0b: u8x16 = p0b .& mask15
        let hi0b: u8x16 = p0b ./ div16

        // Dequant and dot: lo nibbles (dims 0-15, 16-31), hi nibbles (dims 32-47, 48-63)
        let d0: f32x16 = fma(widen_u8_f32x16(lo0a), vs0, vb0)
        acc = acc + reduce_add(d0 .* load(q_vec, qb0))

        let d1: f32x16 = fma(widen_u8_f32x16(lo0b), vs0, vb0)
        acc = acc + reduce_add(d1 .* load(q_vec, qb0 + 16))

        let d2: f32x16 = fma(widen_u8_f32x16(hi0a), vs0, vb0)
        acc = acc + reduce_add(d2 .* load(q_vec, qb0 + 32))

        let d3: f32x16 = fma(widen_u8_f32x16(hi0b), vs0, vb0)
        acc = acc + reduce_add(d3 .* load(q_vec, qb0 + 48))

        // --- Group 1: dimensions 64-127 ---
        let g1: i32 = g0 + 1
        let vs1: f32x16 = splat(k_scales[g1])
        let vb1: f32x16 = splat(k_biases[g1])
        let wb1: i32 = g1 * 32
        let qb1: i32 = 64

        let p1a: u8x16 = load(k_packed, wb1)
        let lo1a: u8x16 = p1a .& mask15
        let hi1a: u8x16 = p1a ./ div16

        let p1b: u8x16 = load(k_packed, wb1 + 16)
        let lo1b: u8x16 = p1b .& mask15
        let hi1b: u8x16 = p1b ./ div16

        let d4: f32x16 = fma(widen_u8_f32x16(lo1a), vs1, vb1)
        acc = acc + reduce_add(d4 .* load(q_vec, qb1))

        let d5: f32x16 = fma(widen_u8_f32x16(lo1b), vs1, vb1)
        acc = acc + reduce_add(d5 .* load(q_vec, qb1 + 16))

        let d6: f32x16 = fma(widen_u8_f32x16(hi1a), vs1, vb1)
        acc = acc + reduce_add(d6 .* load(q_vec, qb1 + 32))

        let d7: f32x16 = fma(widen_u8_f32x16(hi1b), vs1, vb1)
        acc = acc + reduce_add(d7 .* load(q_vec, qb1 + 48))

        scores[t] = acc * rsqrt_d
        t = t + 1
    }
}
```

**Step 2: Compile the kernel**

```bash
EA=/root/dev/eacompute/target/release/ea
$EA kernels/fused_k_score.ea --lib --avx512 -o src/eakv/lib/libfused_k_score.so
```

Expected: `compiled kernels/fused_k_score.ea -> src/eakv/lib/libfused_k_score.so (shared library, 1 exported: q4_fused_k_score_f32)`

**Step 3: Commit**

```bash
git add kernels/fused_k_score.ea src/eakv/lib/libfused_k_score.so
git commit -m "feat: add fused K-score kernel (AVX-512)"
```

---

### Task 2: V Weighted Sum Kernel

Write the Ea kernel that computes `out_vec[d] = sum_t(weights[t] * dequant(V[t])[d])`.

**Files:**
- Create: `kernels/fused_v_sum.ea`

**Context:**
- Same layout as K-score: 2 groups per position, 64 elements per group
- Instead of dot product, this does scaled accumulation: dequant V values, multiply by the softmax weight for this position, add to output vector
- The output vector is 128 f32s (512 bytes) — fits entirely in L1 cache
- Uses `fma(dequantized, splat(weight), current_output)` for the accumulation — one FMA replaces a multiply + add
- Output buffer must be zeroed before the loop (caller responsibility or kernel zeros it)

**Step 1: Write the kernel**

```ea
// Fused V weighted sum: sum_t(weight[t] * dequant(V[t]))
//
// For each position t:
//   Load packed V bytes, dequant inline, multiply by softmax weight,
//   accumulate into output vector [128].
//
// Output vector (512 bytes) stays in L1 the entire time.
// Caller must zero out_vec before calling.

export func q4_fused_v_sum_f32(
    weights:  *restrict f32,
    v_packed: *restrict u8,
    v_scales: *restrict f32,
    v_biases: *restrict f32,
    out_vec:  *mut f32,
    seq_len:  i32
) {
    let mask15: u8x16 = splat(15)
    let div16: u8x16 = splat(16)

    let mut t: i32 = 0
    while t < seq_len {
        let vw: f32x16 = splat(weights[t])

        // --- Group 0: dimensions 0-63 ---
        let g0: i32 = t * 2
        let vs0: f32x16 = splat(v_scales[g0])
        let vb0: f32x16 = splat(v_biases[g0])
        let wb0: i32 = g0 * 32

        let p0a: u8x16 = load(v_packed, wb0)
        let lo0a: u8x16 = p0a .& mask15
        let hi0a: u8x16 = p0a ./ div16

        let p0b: u8x16 = load(v_packed, wb0 + 16)
        let lo0b: u8x16 = p0b .& mask15
        let hi0b: u8x16 = p0b ./ div16

        // Dequant, scale by weight, accumulate
        let d0: f32x16 = fma(widen_u8_f32x16(lo0a), vs0, vb0)
        store(out_vec, 0, fma(d0, vw, load(out_vec, 0)))

        let d1: f32x16 = fma(widen_u8_f32x16(lo0b), vs0, vb0)
        store(out_vec, 16, fma(d1, vw, load(out_vec, 16)))

        let d2: f32x16 = fma(widen_u8_f32x16(hi0a), vs0, vb0)
        store(out_vec, 32, fma(d2, vw, load(out_vec, 32)))

        let d3: f32x16 = fma(widen_u8_f32x16(hi0b), vs0, vb0)
        store(out_vec, 48, fma(d3, vw, load(out_vec, 48)))

        // --- Group 1: dimensions 64-127 ---
        let g1: i32 = g0 + 1
        let vs1: f32x16 = splat(v_scales[g1])
        let vb1: f32x16 = splat(v_biases[g1])
        let wb1: i32 = g1 * 32

        let p1a: u8x16 = load(v_packed, wb1)
        let lo1a: u8x16 = p1a .& mask15
        let hi1a: u8x16 = p1a ./ div16

        let p1b: u8x16 = load(v_packed, wb1 + 16)
        let lo1b: u8x16 = p1b .& mask15
        let hi1b: u8x16 = p1b ./ div16

        let d4: f32x16 = fma(widen_u8_f32x16(lo1a), vs1, vb1)
        store(out_vec, 64, fma(d4, vw, load(out_vec, 64)))

        let d5: f32x16 = fma(widen_u8_f32x16(lo1b), vs1, vb1)
        store(out_vec, 80, fma(d5, vw, load(out_vec, 80)))

        let d6: f32x16 = fma(widen_u8_f32x16(hi1a), vs1, vb1)
        store(out_vec, 96, fma(d6, vw, load(out_vec, 96)))

        let d7: f32x16 = fma(widen_u8_f32x16(hi1b), vs1, vb1)
        store(out_vec, 112, fma(d7, vw, load(out_vec, 112)))

        t = t + 1
    }
}
```

**Step 2: Compile the kernel**

```bash
$EA kernels/fused_v_sum.ea --lib --avx512 -o src/eakv/lib/libfused_v_sum.so
```

Expected: `compiled kernels/fused_v_sum.ea -> src/eakv/lib/libfused_v_sum.so (shared library, 1 exported: q4_fused_v_sum_f32)`

**Step 3: Commit**

```bash
git add kernels/fused_v_sum.ea src/eakv/lib/libfused_v_sum.so
git commit -m "feat: add fused V weighted sum kernel (AVX-512)"
```

---

### Task 3: Python Bindings and Dispatch

Wire the compiled kernels into the Python API via ctypes.

**Files:**
- Create: `src/eakv/_attention.py`
- Modify: `src/eakv/__init__.py` (add exports)

**Context:**
- Follow the same ctypes pattern used in `src/eakv/_dispatch.py`
- The shared library is at `src/eakv/lib/libfused_k_score.so` and `src/eakv/lib/libfused_v_sum.so`
- The Q4Bundle stores weights as `(n_layers, 2, n_groups_per_layer * 32)` u8, scales/biases as `(n_layers, 2, n_groups_per_layer)` f32
- For a single head, we need to slice into the flat packed array. Per head: `n_groups_per_head = seq_len * head_dim / 64`. With head_dim=128: `n_groups_per_head = seq_len * 2`. Weight bytes per head: `n_groups_per_head * 32`.
- The data within a layer/KV is packed as: head 0 groups, then head 1 groups, etc. (because quantize flattens as `n_heads, seq_len, head_dim` then groups over the flat array). So head h starts at group offset `h * seq_len * 2`.

**Step 1: Write `_attention.py`**

```python
"""Fused attention operations on packed Q4 KV cache."""

import ctypes as _ct
import numpy as np
from numpy.typing import NDArray
from pathlib import Path as _Path

from ._bundle import Q4Bundle

_LIB_DIR = _Path(__file__).parent / "lib"
_P_U8 = _ct.POINTER(_ct.c_uint8)
_P_F32 = _ct.POINTER(_ct.c_float)
_I32 = _ct.c_int32

# Load fused kernel libraries
_k_lib = _ct.CDLL(str(_LIB_DIR / "libfused_k_score.so"))
_k_lib.q4_fused_k_score_f32.argtypes = [_P_F32, _P_U8, _P_F32, _P_F32, _P_F32, _I32]
_k_lib.q4_fused_k_score_f32.restype = None

_v_lib = _ct.CDLL(str(_LIB_DIR / "libfused_v_sum.so"))
_v_lib.q4_fused_v_sum_f32.argtypes = [_P_F32, _P_U8, _P_F32, _P_F32, _P_F32, _I32]
_v_lib.q4_fused_v_sum_f32.restype = None


def _head_slice(bundle: Q4Bundle, layer: int, kv: int, head: int):
    """Extract contiguous weight/scale/bias arrays for a single head."""
    groups_per_head = bundle.seq_len * 2  # head_dim=128, 2 groups per position
    g_start = head * groups_per_head
    g_end = g_start + groups_per_head
    w_start = g_start * 32
    w_end = g_end * 32

    weights = np.ascontiguousarray(bundle.weights[layer, kv, w_start:w_end])
    scales = np.ascontiguousarray(bundle.scales[layer, kv, g_start:g_end])
    biases = np.ascontiguousarray(bundle.biases[layer, kv, g_start:g_end])
    return weights, scales, biases


def attention_scores(
    bundle: Q4Bundle,
    query: NDArray,
    layer: int,
    head: int = 0,
) -> NDArray:
    """Compute attention scores directly from packed Q4 K cache.

    Args:
        bundle: Q4Bundle with quantized KV cache
        query: shape [128] f32 — single query vector
        layer: transformer layer index
        head: KV head index

    Returns:
        scores: shape [seq_len] f32 — raw scores (pre-softmax, scaled by 1/sqrt(d))
    """
    if bundle.head_dim != 128:
        raise ValueError(f"Fused attention requires head_dim=128, got {bundle.head_dim}")
    if query.shape != (128,):
        raise ValueError(f"Query must be shape (128,), got {query.shape}")

    q_vec = np.ascontiguousarray(query, dtype=np.float32)
    k_weights, k_scales, k_biases = _head_slice(bundle, layer, 0, head)
    scores = np.empty(bundle.seq_len, dtype=np.float32)

    _k_lib.q4_fused_k_score_f32(
        q_vec.ctypes.data_as(_P_F32),
        k_weights.ctypes.data_as(_P_U8),
        k_scales.ctypes.data_as(_P_F32),
        k_biases.ctypes.data_as(_P_F32),
        scores.ctypes.data_as(_P_F32),
        _I32(bundle.seq_len),
    )
    return scores


def attention_output(
    bundle: Q4Bundle,
    weights: NDArray,
    layer: int,
    head: int = 0,
) -> NDArray:
    """Compute weighted V sum directly from packed Q4 V cache.

    Args:
        bundle: Q4Bundle with quantized KV cache
        weights: shape [seq_len] f32 — softmax attention weights
        layer: transformer layer index
        head: KV head index

    Returns:
        out: shape [128] f32 — attention output vector
    """
    if bundle.head_dim != 128:
        raise ValueError(f"Fused attention requires head_dim=128, got {bundle.head_dim}")

    w = np.ascontiguousarray(weights, dtype=np.float32)
    v_weights, v_scales, v_biases = _head_slice(bundle, layer, 1, head)
    out_vec = np.zeros(128, dtype=np.float32)

    _v_lib.q4_fused_v_sum_f32(
        w.ctypes.data_as(_P_F32),
        v_weights.ctypes.data_as(_P_U8),
        v_scales.ctypes.data_as(_P_F32),
        v_biases.ctypes.data_as(_P_F32),
        out_vec.ctypes.data_as(_P_F32),
        _I32(bundle.seq_len),
    )
    return out_vec
```

**Step 2: Add exports to `__init__.py`**

Add to `src/eakv/__init__.py`:

```python
from ._attention import attention_scores, attention_output
```

And add `"attention_scores"`, `"attention_output"` to `__all__`.

**Step 3: Commit**

```bash
git add src/eakv/_attention.py src/eakv/__init__.py
git commit -m "feat: add Python API for fused attention (attention_scores, attention_output)"
```

---

### Task 4: Correctness Tests

Test that fused kernels produce the same results as restore-then-compute.

**Files:**
- Create: `tests/test_attention.py`

**Context:**
- The reference path: `restore(bundle)` returns full f32 tensor shaped `(n_layers, 2, n_heads, seq_len, head_dim)`. Then `K[head] @ q_vec / sqrt(128)` gives reference scores. `softmax(scores) @ V[head]` gives reference output.
- The fused path: `attention_scores(bundle, q_vec, layer, head)` and `attention_output(bundle, softmax_weights, layer, head)`.
- Both should match within f32 epsilon because they perform identical arithmetic (same widen, same FMA, same accumulation order within each group).
- However, `reduce_add` accumulation order may differ from NumPy's `@` operator, so use `atol=1e-4` to allow for floating-point reassociation.

**Step 1: Write the tests**

```python
"""Test fused attention kernels against reference restore-then-compute."""

import numpy as np
import pytest

from eakv._quantize import quantize
from eakv._restore import dequantize, restore
from eakv._attention import attention_scores, attention_output


def _make_kv_cache(n_layers=1, n_heads=4, seq_len=64, head_dim=128, seed=42):
    rng = np.random.default_rng(seed)
    return rng.standard_normal((n_layers, 2, n_heads, seq_len, head_dim)).astype(np.float32)


def _softmax(x):
    e = np.exp(x - np.max(x))
    return e / e.sum()


class TestFusedKScore:
    def test_basic_correctness(self):
        """Fused K-score matches restore-then-dot."""
        kv = _make_kv_cache()
        bundle = quantize(kv)
        restored = dequantize(bundle)

        q_vec = np.random.default_rng(99).standard_normal(128).astype(np.float32)

        # Reference: restore then dot
        k_head = restored[0, 0, 0]  # (seq_len, head_dim)
        ref_scores = k_head @ q_vec / np.sqrt(128.0)

        # Fused
        fused_scores = attention_scores(bundle, q_vec, layer=0, head=0)

        np.testing.assert_allclose(fused_scores, ref_scores, atol=1e-4)

    def test_all_heads(self):
        """Fused K-score correct for all heads."""
        kv = _make_kv_cache(n_heads=4)
        bundle = quantize(kv)
        restored = dequantize(bundle)
        q_vec = np.random.default_rng(99).standard_normal(128).astype(np.float32)

        for head in range(4):
            ref = restored[0, 0, head] @ q_vec / np.sqrt(128.0)
            fused = attention_scores(bundle, q_vec, layer=0, head=head)
            np.testing.assert_allclose(fused, ref, atol=1e-4,
                                       err_msg=f"head {head} mismatch")

    def test_longer_sequence(self):
        """Works with seq_len=512."""
        kv = _make_kv_cache(seq_len=512)
        bundle = quantize(kv)
        restored = dequantize(bundle)
        q_vec = np.random.default_rng(99).standard_normal(128).astype(np.float32)

        ref = restored[0, 0, 0] @ q_vec / np.sqrt(128.0)
        fused = attention_scores(bundle, q_vec, layer=0, head=0)
        np.testing.assert_allclose(fused, ref, atol=1e-4)

    def test_multiple_layers(self):
        """Works across layers."""
        kv = _make_kv_cache(n_layers=4)
        bundle = quantize(kv)
        restored = dequantize(bundle)
        q_vec = np.random.default_rng(99).standard_normal(128).astype(np.float32)

        for layer in range(4):
            ref = restored[layer, 0, 0] @ q_vec / np.sqrt(128.0)
            fused = attention_scores(bundle, q_vec, layer=layer, head=0)
            np.testing.assert_allclose(fused, ref, atol=1e-4,
                                       err_msg=f"layer {layer} mismatch")


class TestFusedVSum:
    def test_basic_correctness(self):
        """Fused V-sum matches restore-then-matmul."""
        kv = _make_kv_cache()
        bundle = quantize(kv)
        restored = dequantize(bundle)

        # Generate random softmax weights
        raw = np.random.default_rng(77).standard_normal(64).astype(np.float32)
        w = _softmax(raw)

        # Reference: restore then weighted sum
        v_head = restored[0, 1, 0]  # (seq_len, head_dim)
        ref_out = w @ v_head  # (head_dim,)

        # Fused
        fused_out = attention_output(bundle, w, layer=0, head=0)

        np.testing.assert_allclose(fused_out, ref_out, atol=1e-4)

    def test_all_heads(self):
        """Fused V-sum correct for all heads."""
        kv = _make_kv_cache(n_heads=4)
        bundle = quantize(kv)
        restored = dequantize(bundle)

        raw = np.random.default_rng(77).standard_normal(64).astype(np.float32)
        w = _softmax(raw)

        for head in range(4):
            ref = w @ restored[0, 1, head]
            fused = attention_output(bundle, w, layer=0, head=head)
            np.testing.assert_allclose(fused, ref, atol=1e-4,
                                       err_msg=f"head {head} mismatch")

    def test_zero_weights(self):
        """Zero weights produce zero output."""
        kv = _make_kv_cache()
        bundle = quantize(kv)

        w = np.zeros(64, dtype=np.float32)
        fused_out = attention_output(bundle, w, layer=0, head=0)
        np.testing.assert_allclose(fused_out, np.zeros(128), atol=1e-7)

    def test_single_position_weight(self):
        """Weight on single position matches that position's dequantized V."""
        kv = _make_kv_cache()
        bundle = quantize(kv)
        restored = dequantize(bundle)

        w = np.zeros(64, dtype=np.float32)
        w[7] = 1.0  # all weight on position 7

        ref = restored[0, 1, 0, 7]  # V[head=0, pos=7] (head_dim,)
        fused = attention_output(bundle, w, layer=0, head=0)
        np.testing.assert_allclose(fused, ref, atol=1e-4)


class TestAttentionValidation:
    def test_wrong_head_dim(self):
        """Rejects non-128 head_dim."""
        kv = _make_kv_cache(head_dim=64)
        bundle = quantize(kv)
        q_vec = np.zeros(128, dtype=np.float32)
        with pytest.raises(ValueError, match="head_dim=128"):
            attention_scores(bundle, q_vec, layer=0)

    def test_wrong_query_shape(self):
        """Rejects wrong query shape."""
        kv = _make_kv_cache()
        bundle = quantize(kv)
        q_vec = np.zeros(64, dtype=np.float32)
        with pytest.raises(ValueError, match="shape"):
            attention_scores(bundle, q_vec, layer=0)
```

**Step 2: Run tests**

```bash
python3 -m pytest tests/test_attention.py -v
```

Expected: All tests pass.

**Step 3: Commit**

```bash
git add tests/test_attention.py
git commit -m "test: add fused attention correctness tests"
```

---

### Task 5: Benchmark

Measure fused vs. restore-then-compute for realistic sizes.

**Files:**
- Create: `benchmarks/bench_attention.py`

**Context:**
- Baseline: `restore(bundle, layers=layer, heads=head)` to get f32 K, then `K @ q_vec / sqrt(d)`. This does a full memory pass to write f32, then reads it again for the dot product.
- Fused: `attention_scores(bundle, q_vec, layer, head)`. One memory pass over packed Q4 bytes.
- Expected gain: ~1.5-2x because memory traffic roughly halves (no intermediate f32 buffer).
- Test with seq_len=2048 (typical inference), one head at a time.

**Step 1: Write the benchmark**

```python
"""Benchmark: fused attention vs restore-then-compute."""

import sys
import time
import numpy as np

sys.path.insert(0, "src")
from eakv._quantize import quantize
from eakv._restore import dequantize
from eakv._attention import attention_scores, attention_output


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
    avg = np.mean(times) * 1e6  # microseconds
    std = np.std(times) * 1e6
    print(f"  {name:<35s} {avg:>8.0f} us +/- {std:>6.0f}")
    return avg


def main():
    configs = [
        ("7B-like", 1, 8, 2048, 128),
        ("7B-like (long)", 1, 8, 8192, 128),
    ]

    for name, nl, nh, sl, hd in configs:
        print(f"\n{'='*60}")
        print(f"{name}: {nh} heads, {sl} seq, {hd} dim (single layer, single head)")

        kv = np.random.default_rng(42).standard_normal(
            (nl, 2, nh, sl, hd)
        ).astype(np.float32)
        bundle = quantize(kv)
        restored = dequantize(bundle)

        q_vec = np.random.default_rng(99).standard_normal(hd).astype(np.float32)

        # --- K scores ---
        print("\n  K-score (query @ K^T / sqrt(d)):")

        # Baseline: restore then dot
        k_head = restored[0, 0, 0].copy()
        t_base = bench("Baseline (numpy matmul)",
                       lambda: k_head @ q_vec / np.sqrt(128.0))

        # Fused
        t_fused = bench("Fused kernel",
                        lambda: attention_scores(bundle, q_vec, layer=0, head=0))

        print(f"  Speedup: {t_base / t_fused:.2f}x")

        # --- V weighted sum ---
        print("\n  V-sum (softmax @ V):")
        raw = np.random.default_rng(77).standard_normal(sl).astype(np.float32)
        w = _softmax(raw)
        v_head = restored[0, 1, 0].copy()

        t_base = bench("Baseline (numpy matmul)",
                       lambda: w @ v_head)

        t_fused = bench("Fused kernel",
                        lambda: attention_output(bundle, w, layer=0, head=0))

        print(f"  Speedup: {t_base / t_fused:.2f}x")


if __name__ == "__main__":
    main()
```

**Step 2: Run the benchmark**

```bash
python3 benchmarks/bench_attention.py
```

**Step 3: Commit**

```bash
git add benchmarks/bench_attention.py
git commit -m "bench: add fused attention benchmark vs restore-then-compute"
```

---

### Task 6: Update Build Script

Add the fused kernels to `build_kernels.sh` so they are built alongside the dequantize kernels.

**Files:**
- Modify: `build_kernels.sh:75-90` (add after the multi-ISA section)

**Step 1: Add to build script**

Append before the final echo line:

```bash
# Fused attention kernels (AVX-512 only)
echo "  Compiling fused attention kernels..."

"$EA" "$KERNEL_DIR/fused_k_score.ea" --lib --avx512 -o "$LIB_DIR/${PREFIX}fused_k_score${EXT}"
echo "    -> fused_k_score (AVX-512, fused K dot product)"

"$EA" "$KERNEL_DIR/fused_v_sum.ea" --lib --avx512 -o "$LIB_DIR/${PREFIX}fused_v_sum${EXT}"
echo "    -> fused_v_sum (AVX-512, fused V weighted sum)"
```

**Step 2: Run the build script to verify**

```bash
bash build_kernels.sh
```

Expected: All kernels build including the two new fused kernels.

**Step 3: Commit**

```bash
git add build_kernels.sh
git commit -m "build: add fused attention kernels to build script"
```
