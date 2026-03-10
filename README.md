# eakv

Q4 KV cache quantization for LLM inference. C library + CLI tool + Python package.

Compresses KV caches to ~16% of FP32 size (6.4x) using Q4_1 quantization with custom AVX-512 SIMD kernels written in [Eä](https://github.com/petlukk/eacompute). Runs attention directly on compressed data — no decompression needed.

## Performance

Measured on a single core (median of 20 runs):

| Operation | 8 heads, 2K seq | 8 heads, 8K seq |
|---|---|---|
| K-score (1 layer, all heads) | 212 us | 882 us |
| V-output (1 layer, all heads) | 114 us | 486 us |
| **Full attention (1 layer)** | **326 us** | **1368 us** |

K-score throughput: ~5 GB/s on packed Q4 data (memory bandwidth bound).

For a 32-layer 7B model at 2K context: **~10 ms total** for all KV attention.

### GQA speedups

Loop-flipped kernels dequantize K/V once and reuse across grouped query heads:

| Config | K-score speedup | V-sum speedup |
|---|---|---|
| 32Q / 8KV (4:1) | 2.08x | 3.71x |
| 8Q / 2KV (4:1) | 1.68x | 3.43x |
| 4Q / 2KV (2:1) | 1.20x | 2.34x |

### Compression

Q4_1 with group size 64. Each group of 64 values → 32 packed bytes + scale (f32) + bias (f32) = 40 bytes.

| Model | Context | FP32 | Q4 (eakv) | Ratio |
|---|---|---|---|---|
| 7B (8 KV heads) | 2K | 16 MB/layer | 2.5 MB/layer | 6.4x |
| 7B (8 KV heads) | 8K | 64 MB/layer | 10 MB/layer | 6.4x |
| 7B (32 layers) | 2K | 512 MB total | 80 MB total | 6.4x |

## C library (libeakv)

### Build

```bash
./build_kernels.sh    # compile Eä SIMD kernels (.ea -> .o)
make                  # build libeakv.a + libeakv.so + eakv CLI
make test             # run C tests (12 tests)
make bench            # run benchmarks
```

Requires the [Eä compiler](https://github.com/petlukk/eacompute) for kernel compilation. Set `EA=/path/to/ea` if not in PATH.

### C API

```c
#include "eakv.h"

// Create cache (pre-allocates for max_seq_len)
eakv_cache_t *cache = eakv_cache_create(32, 8, 128, 4096);

// Bulk-quantize f32 KV data: [layer][kv][head][pos][dim]
eakv_cache_load_raw(cache, kv_data, seq_len);

// Attention directly on Q4 data (MHA or GQA)
float queries[32 * 128], scores[32 * 4096], output[32 * 128];
eakv_attention_scores(cache, queries, layer, 32, 8, scores);
// ... softmax ...
eakv_attention_output(cache, weights, layer, 32, 8, output);

// File I/O
eakv_cache_save(cache, "cache.eakv");
eakv_cache_load("cache.eakv", &cache);

eakv_cache_free(cache);
```

See `include/eakv.h` for the full API.

### CLI

```bash
eakv inspect cache.eakv    # show metadata, sizes, compression ratio
eakv validate cache.eakv   # check for NaN/corruption via SIMD kernel
```

## Python package

### Install

```bash
pip install eakv
```

### Quick start

```python
import numpy as np
import eakv

# Quantize: (n_layers, 2, n_heads, seq_len, head_dim)
kv_cache = np.random.randn(32, 2, 8, 2048, 128).astype(np.float32)
bundle = eakv.quantize(kv_cache)

# Save / load
eakv.save(bundle, "cache.eakv")
bundle = eakv.load("cache.eakv")

# GQA attention (auto-dispatches MHA when n_q == n_kv)
queries = np.random.randn(32, 128).astype(np.float32)
scores = eakv.attention_scores_gqa(bundle, queries, layer=0,
                                    n_q_heads=32, n_kv_heads=8)

# Validate integrity
eakv.validate(bundle)
```

## Project structure

```
include/                     C public header
  eakv.h                       Single-header API (opaque handle)

src/                         C library source
  internal.h                   Private struct + kernel externs
  cache.c                      Cache lifecycle, load_raw (bulk quantize)
  attention.c                  MHA/GQA attention dispatch
  io.c                         .eakv binary format save/load
  cli.c                        CLI tool (inspect, validate)

src/eakv/                    Python library
  _bundle.py                   Q4Bundle dataclass
  _quantize.py                 quantize() API
  _restore.py                  dequantize() / restore() with partial select
  _attention.py                Fused attention (MHA + GQA with auto-dispatch)
  _dispatch.py                 Runtime ISA detection (AVX-512 > AVX2 > SSE)
  _io.py                       Binary .eakv format + mmap

kernels/                     Eä SIMD kernel source (.ea)
  quantize_simd.ea             Q4_1 quantization (split lo/hi nibble packing)
  dequantize_{simd,avx2,avx512}.ea   Multi-ISA dequantization
  fused_k_score.ea             Fused query @ K^T (single + multi-head)
  fused_v_sum.ea               Fused weights @ V (single + multi-head)
  fused_k_score_gqa.ea         GQA loop-flipped K-score + V-sum
  fused_attention.ea           Fused softmax+attention (2-pass)
  validate.ea                  NaN/negative scale detection

tests/                       C tests (12) + Python tests (60)
benchmarks/                  C + Python benchmarks
```

## Requirements

- x86-64 CPU (AVX-512 recommended, SSE minimum for Python package)
- C library: GCC, Make, Eä compiler (for kernel compilation)
- Python package: Python >= 3.9, NumPy >= 1.21
