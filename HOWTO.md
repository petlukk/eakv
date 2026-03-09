# How To Use eakv

## 1. Quantize a KV cache

eakv expects a 5D numpy array shaped `(n_layers, 2, n_heads, seq_len, head_dim)` where axis 1 is `[K, V]`. This matches the layout used by most transformer implementations.

```python
import numpy as np
import eakv

# From your model's KV cache (typically extracted from model internals)
# k_cache: (n_layers, n_heads, seq_len, head_dim)
# v_cache: (n_layers, n_heads, seq_len, head_dim)
kv = np.stack([k_cache, v_cache], axis=1)  # -> (n_layers, 2, n_heads, seq_len, head_dim)

bundle = eakv.quantize(kv)
print(f"Compressed to {bundle.compression_ratio:.0%} of original")
```

The input must be float32 or float16. `head_dim * seq_len * n_heads` must be a multiple of 64 (the quantization group size). This is true for all standard models (head_dim is always 64 or 128).

## 2. Save and load

```python
# Save to disk
eakv.save(bundle, "kv_cache.eakv")

# Save with zstd compression (requires: pip install zstandard)
eakv.save(bundle, "kv_cache.eakv", compression="zstd")

# Load
bundle = eakv.load("kv_cache.eakv")
```

The `.eakv` binary format stores metadata (layer count, head count, dimensions, original dtype) in a 512-byte header followed by 64-byte-aligned data arrays.

## 3. Restore (decompress) the KV cache

```python
# Full restore to FP32
kv_restored = eakv.dequantize(bundle)  # -> (n_layers, 2, n_heads, seq_len, head_dim)

# Partial restore: specific layers, heads, or token ranges
layer_0 = eakv.restore(bundle, layers=0)
heads_01 = eakv.restore(bundle, heads=[0, 1])
last_64 = eakv.restore(bundle, tokens=-64)
combined = eakv.restore(bundle, layers=range(4), heads=0, tokens=-128)

# Restore to FP16
kv_f16 = eakv.restore(bundle, out_dtype="float16")
```

Partial restore still dequantizes full layers internally, then slices. It avoids allocating the full output array.

## 4. Fused attention (single head)

Compute attention directly on Q4 data. No decompression step, no intermediate arrays.

```python
query = np.random.randn(128).astype(np.float32)  # head_dim must be 128

# K-scores: dot(query, K[t]) / sqrt(head_dim) for each position t
scores = eakv.attention_scores(bundle, query, layer=0, head=0)
# -> shape (seq_len,), f32

# Softmax
def softmax(x):
    e = np.exp(x - x.max())
    return e / e.sum()
weights = softmax(scores)

# V weighted sum: sum_t(weights[t] * V[t])
output = eakv.attention_output(bundle, weights, layer=0, head=0)
# -> shape (128,), f32
```

This is the core decode-time attention operation. The fused kernels dequantize each position into SIMD registers and immediately multiply, never writing intermediate f32 values to memory.

## 5. Fused attention (all heads, MHA)

For multiple heads, use the multi-head variants to avoid per-head Python overhead:

```python
n_heads = 8
queries = np.random.randn(n_heads, 128).astype(np.float32)

# All K-scores in one kernel call
all_scores = eakv.attention_scores_multi(bundle, queries, layer=0, n_heads=n_heads)
# -> shape (n_heads, seq_len), f32

# Softmax each head
all_weights = np.array([softmax(all_scores[h]) for h in range(n_heads)])

# All V weighted sums in one kernel call
all_outputs = eakv.attention_output_multi(bundle, all_weights, layer=0, n_heads=n_heads)
# -> shape (n_heads, 128), f32
```

At 2048 tokens with 8 heads, the multi-head kernel is ~2x faster than calling the single-head version in a Python loop.

## 6. GQA attention (Grouped Query Attention)

For models with fewer KV heads than query heads (LLaMA 2/3, Mistral, etc.):

```python
n_q_heads = 32   # query heads
n_kv_heads = 8   # KV heads (4:1 ratio)

queries = np.random.randn(n_q_heads, 128).astype(np.float32)

# K-scores: dequantizes K once per token, reuses across grouped Q heads
scores = eakv.attention_scores_gqa(bundle, queries, layer=0,
                                    n_q_heads=n_q_heads, n_kv_heads=n_kv_heads)
# -> shape (n_q_heads, seq_len), f32

# Softmax each head
weights = np.array([softmax(scores[h]) for h in range(n_q_heads)])

# V-sum: 2-head paired accumulation for efficient register use
output = eakv.attention_output_gqa(bundle, weights, layer=0,
                                    n_q_heads=n_q_heads, n_kv_heads=n_kv_heads)
# -> shape (n_q_heads, 128), f32
```

The GQA functions auto-dispatch: when `n_q_heads == n_kv_heads` (MHA), they route to the faster MHA kernels internally. You can use `attention_scores_gqa` / `attention_output_gqa` as the universal entry point.

**Speedups vs naive (separate kernel per Q head):**

| GQA ratio | K-score | V-sum |
|---|---|---|
| 4:1 (32Q/8KV) | 2.08x | 3.71x |
| 4:1 (8Q/2KV) | 1.68x | 3.43x |
| 2:1 (4Q/2KV) | 1.20x | 2.34x |
| 1:1 (MHA) | auto-dispatch to MHA kernel | auto-dispatch to MHA kernel |

## 7. Memory-mapped access

For large caches that don't fit in RAM:

```python
with eakv.open_mmap("kv_cache.eakv") as bundle:
    # bundle.weights, .scales, .biases are mmap views (not loaded into RAM)
    scores = eakv.attention_scores(bundle, query, layer=0, head=0)
    # Only the accessed pages are loaded from disk
```

## 8. Validate a bundle

Check for NaN scales/biases or negative scales (corruption indicators):

```python
# Python API
eakv.validate(bundle)  # raises ValueError on corruption

# CLI
# eakv validate kv_cache.eakv
```

## 9. Inspect a file

```bash
$ eakv inspect kv_cache.eakv
File:          kv_cache.eakv
File size:     167.2 MB
Original size: 536.9 MB (float32)
Compression:   31.1%
Layers:        32
Heads:         8
Seq length:    2048
Head dim:      128
Groups/layer:  32768
Quant scheme:  Q4_1 (group size 64)
```

## 10. ISA detection

eakv auto-detects your CPU and loads the fastest available kernel:

```python
print(eakv.get_isa())  # 'avx512', 'avx2', or 'sse'
```

AVX-512 is recommended. AVX2 and SSE work but are slower for dequantization. The fused attention and GQA kernels require AVX-512.

## 11. Build kernels from source

If you need to rebuild the SIMD kernels (after modifying `.ea` source files):

```bash
# Set EA to the Eä compiler path (or add to PATH)
export EA=$HOME/dev/eacompute/target/release/ea

./build_kernels.sh
```

This compiles 9 kernel files into 10 shared libraries and generates 2 Python binding files.
