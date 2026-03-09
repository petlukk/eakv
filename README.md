# eakv

Fast Q4 KV cache quantization for LLM inference. Compresses KV caches to ~31% of FP16 size (3.2x) and computes attention directly on compressed data without decompressing.

Built on [Eä](https://github.com/petlukk/eacompute) SIMD kernels (AVX-512 / AVX2 / SSE).

## What it does

| Operation | Description |
|---|---|
| `quantize` | FP32/FP16 KV cache &rarr; Q4_1 packed format (group size 64) |
| `dequantize` | Q4_1 &rarr; FP32 (full or partial: by layer, head, token range) |
| `attention_scores_gqa` | Query @ K^T / sqrt(d) directly on Q4 bytes, GQA-aware |
| `attention_output_gqa` | Softmax weights @ V directly on Q4 bytes, GQA-aware |
| `save` / `load` | Binary `.eakv` format with optional zstd compression |
| `open_mmap` | Memory-mapped access for large caches |
| `validate` | SIMD integrity check (NaN, negative scale detection) |

## Install

```bash
pip install eakv
```

Or from source:

```bash
pip install -e .
```

Pre-built wheels include all SIMD kernel libraries. No compiler needed.

## Compression

Q4_1 with group size 64. Each group of 64 values is stored as 32 packed bytes + 1 scale (f32) + 1 bias (f32).

| Model | Context | FP16 | Q4 (eakv) | Saved |
|---|---|---|---|---|
| Llama-2-7B (8 KV heads) | 4K | 0.54 GB | 0.17 GB | 0.37 GB |
| Llama-2-7B (8 KV heads) | 8K | 1.07 GB | 0.34 GB | 0.74 GB |

## Fused attention

The fused kernels compute attention scores and weighted V sums directly from Q4 packed bytes. No intermediate FP32 arrays are materialized. Each sequence position is dequantized into registers, multiplied, and accumulated in a single pass.

### Multi-head attention (MHA)

Processes all heads in one kernel call, eliminating per-head Python/ctypes overhead:

| seq_len | Per-head loop (8 heads) | Multi-head kernel | Speedup |
|---|---|---|---|
| 2048 | 850 us | 413 us | 2.1x |
| 4096 | 1313 us | 902 us | 1.5x |
| 8192 | 1729 us | 1574 us | 1.1x |

### Grouped Query Attention (GQA)

Loop-flipped kernels that dequantize K/V once and reuse across all query heads sharing a KV head. Auto-dispatches to MHA kernels when `n_q_heads == n_kv_heads`.

**K-score** — dequantize K once per token, dot with all grouped Q heads:

| Config | Speedup vs naive |
|---|---|
| 32Q / 8KV (4:1) | 2.08x |
| 8Q / 2KV (4:1) | 1.68x |
| 4Q / 2KV (2:1) | 1.20x |

**V-sum** — 2-head paired accumulation, 24/32 ZMM registers:

| Config | Speedup vs naive |
|---|---|
| 32Q / 8KV (4:1) | 3.71x |
| 8Q / 2KV (4:1) | 3.43x |
| 4Q / 2KV (2:1) | 2.34x |

## Quick start

```python
import numpy as np
import eakv

# Quantize a KV cache: shape (n_layers, 2, n_heads, seq_len, head_dim)
kv_cache = np.random.randn(32, 2, 8, 2048, 128).astype(np.float32)
bundle = eakv.quantize(kv_cache)

# Save / load
eakv.save(bundle, "cache.eakv")
bundle = eakv.load("cache.eakv")

# Restore to FP32 (full or partial)
restored = eakv.dequantize(bundle)
partial = eakv.restore(bundle, layers=0, heads=[0, 1], tokens=-64)

# GQA attention (auto-dispatches MHA when n_q == n_kv)
n_q_heads, n_kv_heads = 32, 8
queries = np.random.randn(n_q_heads, 128).astype(np.float32)
scores = eakv.attention_scores_gqa(bundle, queries, layer=0,
                                    n_q_heads=n_q_heads, n_kv_heads=n_kv_heads)

import scipy.special
weights = scipy.special.softmax(scores, axis=1)
output = eakv.attention_output_gqa(bundle, weights, layer=0,
                                    n_q_heads=n_q_heads, n_kv_heads=n_kv_heads)
# -> shape (32, 128), f32

# Validate integrity
eakv.validate(bundle)
```

## CLI

```bash
eakv inspect cache.eakv    # show metadata, sizes, compression ratio
eakv validate cache.eakv   # check for NaN/corruption
```

## Tests

```bash
pip install -e ".[dev]"
pytest tests/ -v
```

60 tests covering quantize/restore roundtrip, fused attention correctness (MHA, GQA, all heads/layers/sequence lengths), I/O, validation, and partial restore.

## Benchmarks

```bash
python3 benchmarks/bench_attention.py   # fused attention vs restore+matmul
python3 benchmarks/bench_accuracy.py    # quantization error analysis
python3 benchmarks/bench_roundtrip.py   # quantize/save/load/restore throughput
```

## Project structure

```
kernels/                     Eä SIMD kernel source (.ea)
  quantize_simd.ea             Q4_1 quantization (split lo/hi nibble packing)
  dequantize_simd.ea           SSE dequantization (f32x4)
  dequantize_avx2.ea           AVX2 dequantization (f32x8)
  dequantize_avx512.ea         AVX-512 dequantization (f32x16)
  fused_k_score.ea             Fused query @ K^T (single + multi-head)
  fused_v_sum.ea               Fused weights @ V (single + multi-head)
  fused_k_score_gqa.ea         GQA loop-flipped K-score + V-sum (2-head pairing)
  fused_attention.ea           Experimental fused softmax+attention
  validate.ea                  NaN/negative scale detection

src/eakv/                    Python library
  _bundle.py                   Q4Bundle dataclass
  _quantize.py                 quantize() API
  _restore.py                  dequantize() / restore() with partial select
  _attention.py                Fused attention API (MHA + GQA with auto-dispatch)
  _dispatch.py                 Runtime ISA detection (AVX-512 > AVX2 > SSE)
  _io.py                       Binary .eakv format + mmap
  _ops.py                      Kernel function re-exports
  cli.py                       eakv inspect / validate commands

tests/                       60 tests
benchmarks/                  Performance benchmarks
```

## Requirements

- Python >= 3.9
- NumPy >= 1.21
- x86-64 CPU (SSE minimum, AVX-512 recommended)
- Optional: `zstandard` for compressed .eakv files
