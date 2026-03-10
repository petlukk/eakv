# How To Use eakv

eakv has two interfaces: a C library (`libeakv`) and a Python package. Both share the same Eä SIMD kernels and `.eakv` file format.

## C Library

### Build

```bash
./build_kernels.sh    # compile .ea kernels -> .o object files + .so shared libs
make                  # build libeakv.a, libeakv.so, eakv CLI
make test             # 12 tests
make bench            # performance benchmarks
```

### 1. Create a cache

```c
#include "eakv.h"

// Pre-allocates for max_seq_len tokens. No realloc during use.
eakv_cache_t *cache = eakv_cache_create(
    32,    // n_layers
    8,     // n_kv_heads
    128,   // head_dim
    4096   // max_seq_len
);
```

`n_kv_heads * head_dim` must be a multiple of 64 (the quantization group size). This is true for all standard models.

### 2. Load data

```c
// f32 array shaped [n_layers][2][n_kv_heads][seq_len][head_dim]
// Axis 1: 0=K, 1=V
float *kv_data = /* your KV cache */;
int rc = eakv_cache_load_raw(cache, kv_data, seq_len);
// rc == EAKV_OK on success
```

### 3. Run attention

```c
int n_q_heads = 32, n_kv_heads = 8;  // GQA 4:1
int seq_len = eakv_cache_seq_len(cache);

float queries[32 * 128];   // your query vectors
float scores[32 * 4096];   // output: pre-softmax scores
float weights[32 * 4096];  // input: post-softmax weights
float output[32 * 128];    // output: attention vectors

// K-scores: Q @ K^T / sqrt(d), directly on Q4 bytes
eakv_attention_scores(cache, queries, layer, n_q_heads, n_kv_heads, scores);

// ... apply softmax to scores -> weights ...

// V weighted sum: weights @ V, directly on Q4 bytes
eakv_attention_output(cache, weights, layer, n_q_heads, n_kv_heads, output);
```

MHA and GQA use the same functions. When `n_q_heads == n_kv_heads`, the library dispatches to the faster MHA kernel automatically.

### 4. Save and load

```c
// Save
eakv_cache_save(cache, "session.eakv");

// Load
eakv_cache_t *loaded = NULL;
eakv_cache_load("session.eakv", &loaded);

// Clean up
eakv_cache_free(cache);
eakv_cache_free(loaded);
```

Files are byte-compatible between the C library and Python package.

### 5. CLI

```bash
$ eakv inspect session.eakv
session.eakv
  layers:     32
  kv_heads:   8
  head_dim:   128
  seq_len:    2048
  file_size:  80.0 MB
  orig_size:  512.0 MB
  ratio:      6.4x

$ eakv validate session.eakv
session.eakv: ok (32 layers checked)
```

### Error handling

All functions returning `int` use these codes:

| Code | Meaning |
|---|---|
| `EAKV_OK` (0) | Success |
| `EAKV_ERR_IO` (-1) | File open/read/write failed |
| `EAKV_ERR_FORMAT` (-2) | Bad magic, version, or truncated file |
| `EAKV_ERR_ALLOC` (-3) | malloc failed |
| `EAKV_ERR_INVALID` (-4) | Bad parameters (null, out of range) |

### Linking

```bash
# Static
gcc myapp.c build/libeakv.a -lm -o myapp

# Shared
gcc myapp.c -Lbuild -leakv -lm -o myapp
```

---

## Python Package

### Install

```bash
pip install eakv
```

Pre-built wheels include all SIMD kernel libraries. No compiler needed.

### 1. Quantize a KV cache

```python
import numpy as np
import eakv

# From your model's KV cache
# k_cache: (n_layers, n_heads, seq_len, head_dim)
# v_cache: (n_layers, n_heads, seq_len, head_dim)
kv = np.stack([k_cache, v_cache], axis=1)
# -> (n_layers, 2, n_heads, seq_len, head_dim)

bundle = eakv.quantize(kv)
```

Input must be float32 or float16.

### 2. Save and load

```python
eakv.save(bundle, "cache.eakv")
bundle = eakv.load("cache.eakv")

# Memory-mapped access for large caches
with eakv.open_mmap("cache.eakv") as bundle:
    scores = eakv.attention_scores(bundle, query, layer=0, head=0)
```

### 3. Restore (decompress)

```python
# Full restore to f32
restored = eakv.dequantize(bundle)

# Partial restore: layer, head, or token slicing
partial = eakv.restore(bundle, layers=0, heads=[0, 1], tokens=-64)
```

### 4. Fused attention

```python
# Single head
scores = eakv.attention_scores(bundle, query, layer=0, head=0)
output = eakv.attention_output(bundle, weights, layer=0, head=0)

# All heads (one kernel call)
scores = eakv.attention_scores_multi(bundle, queries, layer=0, n_heads=8)
output = eakv.attention_output_multi(bundle, all_weights, layer=0, n_heads=8)

# GQA (auto-dispatches MHA when n_q == n_kv)
scores = eakv.attention_scores_gqa(bundle, queries, layer=0,
                                    n_q_heads=32, n_kv_heads=8)
output = eakv.attention_output_gqa(bundle, weights, layer=0,
                                    n_q_heads=32, n_kv_heads=8)
```

### 5. Validate and inspect

```python
eakv.validate(bundle)  # raises ValueError on corruption
print(eakv.get_isa())  # 'avx512', 'avx2', or 'sse'
```

```bash
eakv inspect cache.eakv
eakv validate cache.eakv
```

### 6. Build kernels from source

```bash
export EA=$HOME/dev/eacompute/target/release/ea
./build_kernels.sh
```
