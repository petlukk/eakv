# eakv v0.1 Design

**Date**: 2026-03-09
**Status**: Approved

## What it is

A pip-installable Python library (`pip install eakv`) backed by SIMD kernels written in Ea, providing fast Q4 KV cache quantization, save, and restore for LLM inference. Targets llama.cpp integration first.

Core value prop: ~25x faster session resumption by persisting KV caches in compressed Q4 format instead of recomputing prefill.

## Architecture

```
eakv/
├── kernels/
│   ├── quantize.ea             # FP16/FP32 -> Q4_1 (scale+bias, group 64)
│   ├── dequantize.ea           # Q4_1 -> FP16/FP32, with partial restore support
│   ├── pack.ea                 # Bit-packing and memory layout transforms
│   ├── append.ea               # Append new tokens to Q4 KV without re-quantization
│   └── validate.ea             # NaN/overflow/corruption checks on Q4 data
├── build_kernels.sh            # ea compiler -> .so + .h + .ea.json
├── src/eakv/
│   ├── __init__.py             # Public API
│   ├── _loader.py              # ctypes dynamic lib loading (eavec pattern)
│   ├── _ops.py                 # Low-level kernel wrappers
│   ├── _quantize.py            # quantize(kv_tensor) -> Q4Bundle
│   ├── _restore.py             # restore(Q4Bundle, layers, tokens, heads) -> tensor
│   ├── _io.py                  # save/load/mmap Q4 bundles to disk
│   ├── _llama.py               # llama.cpp import/export helpers
│   └── lib/                    # Pre-compiled .so files
├── include/
│   └── eakv.h                  # Generated C header for native integration
├── pyproject.toml
├── tests/
│   ├── test_quantize.py
│   ├── test_restore.py
│   ├── test_io.py
│   ├── test_append.py
│   ├── test_validate.py
│   └── test_llama.py
└── benchmarks/
    └── bench_prefill_skip.py   # The benchmark that matters: prefill vs restore
```

## Ea Kernels (5 files)

### quantize.ea

Per-group Q4_1 quantization. Group size 64, find min/max per group, compute scale+bias, pack 8 values into one u32.

```
q4_quantize_f16(src, weights_out, scales_out, biases_out, n_groups)
q4_quantize_f32(src, weights_out, scales_out, biases_out, n_groups)
```

### dequantize.ea

Q4_1 -> float restore with partial support. Can dequantize a contiguous range of groups (for layer/token slicing).

```
q4_dequantize_f16(weights, scales, biases, out, n_groups)
q4_dequantize_f32(weights, scales, biases, out, n_groups)
q4_dequantize_range_f16(weights, scales, biases, out, group_start, group_count)
q4_dequantize_range_f32(weights, scales, biases, out, group_start, group_count)
```

### pack.ea

Memory layout utilities for cache-friendly head interleaving.

```
q4_pack_heads(src, dst, n_heads, head_dim, seq_len)
q4_deinterleave_heads(src, dst, n_heads, head_dim, seq_len)
```

### append.ea

Append new token KV to existing Q4 cache without full re-quantization. Quantizes the new token's values and writes them into the correct position in the packed layout.

```
q4_append_token_f16(src_token, weights, scales, biases, position, head_dim)
q4_append_token_f32(src_token, weights, scales, biases, position, head_dim)
```

### validate.ea

Debug/safety kernel. Checks Q4 data for corruption.

```
q4_validate(weights, scales, biases, n_groups) -> i32  # 0 = ok, error code otherwise
```

## Python API

### Core operations

```python
import eakv
import numpy as np

# Quantize KV cache (FP16 or FP32, any shape)
bundle = eakv.quantize(kv_cache)

# Full restore
kv_cache = eakv.dequantize(bundle)

# Partial restore (the key differentiator)
kv = eakv.restore(bundle, layers=range(0, 8))         # first 8 layers only
kv = eakv.restore(bundle, tokens=-256)                  # last 256 tokens
kv = eakv.restore(bundle, layers=[0,1], tokens=-128)    # combined
kv = eakv.restore(bundle, out_dtype="f32")               # control output type

# Append token without re-quantization
eakv.append(bundle, new_kv_token, position=seq_len)

# Validate
eakv.validate(bundle)  # raises on corruption
```

### File I/O with mmap

```python
# Standard save/load
eakv.save(bundle, "session_001.eakv")
bundle = eakv.load("session_001.eakv")

# Memory-mapped (first-class, no copy)
with eakv.open("session_001.eakv") as bundle:
    kv = eakv.restore(bundle, tokens=-256)

# Save with optional zstd compression
eakv.save(bundle, "session_001.eakv", compress="zstd")

# Save with model identity for safety
eakv.save(bundle, "session_001.eakv", model_hash="abc123", tokenizer_hash="def456")
```

### llama.cpp integration

```python
# Import KV cache from llama.cpp session file
bundle = eakv.import_llama("session.bin")

# Export back to llama.cpp format
eakv.export_llama(bundle, "session_restored.bin")
```

## C API

```c
#include "eakv.h"

// Quantize
eakv_q4_quantize_f16(src, weights, scales, biases, n_groups);

// Restore (full)
eakv_q4_dequantize_f16(weights, scales, biases, dst, n_groups);

// Restore (partial range)
eakv_q4_dequantize_range_f16(weights, scales, biases, dst, group_start, group_count);

// Append token
eakv_q4_append_token_f16(src_token, weights, scales, biases, position, head_dim);

// Validate
int status = eakv_q4_validate(weights, scales, biases, n_groups);
```

Link with: `cc -leakv program.c`

## .eakv File Format

Binary format, designed for mmap. All data 64-byte aligned.

### Header (512 bytes, reserved for future versions)

| Field | Type | Description |
|-------|------|-------------|
| magic | u8[4] | `EAKV` |
| version | u16 | Format version (1) |
| quant_scheme | u16 | 0=Q4_1 (extensible to Q4_0, Q8_0) |
| group_size | u32 | 64 (extensible) |
| orig_dtype | u16 | 0=f16, 1=f32 |
| n_layers | u32 | Number of layers |
| n_heads | u32 | Number of KV heads |
| head_dim | u32 | Dimension per head |
| seq_len | u32 | Current sequence length |
| max_seq_len | u32 | Allocated capacity |
| compression | u16 | 0=none, 1=zstd |
| model_hash | u8[32] | Optional model identity (0 if unset) |
| tokenizer_hash | u8[32] | Optional tokenizer identity (0 if unset) |
| checksum | u64 | CRC64 of data section |
| reserved | u8[...] | Pad to 512 bytes |

### Data section

Per-layer, contiguous, 64-byte aligned:

```
[layer 0 K weights][layer 0 K scales][layer 0 K biases]
[layer 0 V weights][layer 0 V scales][layer 0 V biases]
[layer 1 K weights][layer 1 K scales][layer 1 K biases]
[layer 1 V weights][layer 1 V scales][layer 1 V biases]
...
```

Per-layer offsets stored in a layer index table immediately after the header (n_layers * 2 * u64 = K_offset, V_offset per layer). This enables O(1) seeking for partial restore.

### Storage math

8B model (32 layers, 32 KV heads, 4096 seq, 128 head_dim):
- FP16 KV: ~2 GB
- Q4_1 eakv: ~562 MB (28.1% of FP16)
- Per-layer: ~8.8 MB

## SIMD alignment requirements

- Group size: 64 elements (fits in 4x f32x8 or 8x f32x8 loads)
- All buffer allocations: 64-byte aligned
- File data sections: 64-byte aligned (mmap-friendly)
- Hot path stays in L1/L2 cache

## Kernel design notes

### Q4_1 quantization per group (64 elements)

```
1. Load 64 floats (8x f32x8 loads)
2. Reduce to find min, max across all 64
3. scale = (max - min) / 15.0
4. bias = min
5. For each value: q = round((val - bias) / scale), clamp to [0, 15]
6. Pack pairs of 4-bit values into u8, then u8 pairs into u32
7. Store: 16 bytes packed weights + 2 bytes scale + 2 bytes bias = 20 bytes per group
```

### Q4_1 dequantization per group

```
1. Load scale, bias (bf16 -> f32)
2. Load 16 bytes packed weights
3. Unpack 4-bit values (mask + shift)
4. val = q * scale + bias
5. Store 64 floats
```

Both operations are embarrassingly SIMD-parallel across groups.

## What's NOT in v0.1

- Fused Q4 attention (v0.2)
- Agent session management / eviction policies (v0.2)
- Q4_0 / Q8_0 quantization variants (v0.2)
- GPU kernels (CPU SIMD only for v0.1)
- Multi-file sharded caches (v0.2)

## Key benchmark to ship with

```
Prompt length: 4096 tokens
Model: Llama 3.1 8B

Normal resume (recompute prefill): ~15.7s
eakv restore (Q4, full):            ~0.6s
eakv restore (Q4, last 256 tokens): ~0.04s
Speedup: 25-400x
```

This benchmark is the marketing. Ship it prominently.
