# eakv

Q4 KV cache compression for LLM inference. C library with custom AVX-512 SIMD kernels.

Compresses KV caches **~6–13x** vs F16, runs fused attention **~5–8x faster** than f32 dot product — directly on packed Q4 data, no decompression needed.

Supports both MHA and GQA (grouped-query attention) with any head_dim that's a multiple of 64.

## Validated on Real Model Data

Tested with TinyLlama 1.1B (22 layers, 32 Q heads, 4 KV heads, head_dim=64, GQA 8:1) using real KV cache data extracted from llama.cpp:

### Quantization Quality

| Metric | Value |
|---|---|
| SNR | 19.1 dB |
| RMSE | 0.135 |
| Relative RMSE | 11.14% |
| Max element error | 0.576 |

### Attention Accuracy (fused Q4 vs f32 reference)

| Metric | Value |
|---|---|
| Mean absolute error | 0.065 |
| Mean relative error | 15.45% |
| NaN/Inf | 0 |

Scores are close enough that both Q4 and F16 KV caches produce coherent, factually reasonable text output from llama.cpp.

### Speed

Fused Q4 attention kernels vs baselines (single CPU core, median timing):

| Config | Fused Q4 | Dequant+dot | F32 dot | Speedup vs f32 |
|---|---|---|---|---|
| 8H, 128d, 2K seq | 210 us | 1,192 us | 1,051 us | **5.0x** |
| 8H, 128d, 8K seq | 844 us | 5,731 us | 4,615 us | **5.5x** |
| 4H, 64d, 32 seq (TinyLlama real data) | 1 us | — | 8 us | **8x** |

Speedup grows with sequence length due to better Q4 data locality.

### Memory Compression

| Format | Size (7B, 2K seq) | vs F32 | vs F16 |
|---|---|---|---|
| F32 | 16.0 MB | 1.0x | — |
| F16 (llama.cpp default) | 8.0 MB | 2.0x | 1.0x |
| Q4 (eakv) | 2.5 MB | **6.4x** | **3.2x** |

With smaller head_dim models (e.g. TinyLlama, head_dim=64), compression reaches **12.8x vs F16**.

### GQA Speedups

Loop-flipped kernels dequantize K/V once per token and reuse across grouped query heads:

| Config | K-score speedup | V-sum speedup |
|---|---|---|
| 32Q / 8KV (4:1) | 2.08x | 3.71x |
| 8Q / 2KV (4:1) | 1.68x | 3.43x |
| 4Q / 2KV (2:1) | 1.20x | 2.34x |

## How It Works

Each group of 64 f32 values is quantized to 4-bit integers with per-group scale and bias (Q4_1 format). The fused AVX-512 kernels compute `Q @ K^T / sqrt(d)` and `weights @ V` directly on packed nibbles — no intermediate f32 buffer needed.

Key optimizations:
- Query/output vectors held in ZMM registers across all sequence positions
- FMA chains: `fma(widen_u8_f32x16(nibbles), scale, bias)` fused with dot accumulation
- GQA loop flip: outer loop over KV heads, dequantize once, inner loop over Q heads
- 2-position unrolling for instruction-level parallelism

Kernels are written in [Ea](https://github.com/niclas-edenworlds/eacompute), a SIMD-first language that compiles to native x86-64.

## Build

```bash
./build_kernels.sh    # compile Ea SIMD kernels (.ea -> .o)
make                  # build libeakv.a + libeakv.so + eakv CLI
make test             # run C tests
make bench            # run benchmarks
```

Requires the [Ea compiler](https://github.com/niclas-edenworlds/eacompute) for kernel compilation. Set `EA=/path/to/ea` if not in PATH.

## C API

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

See [`include/eakv.h`](include/eakv.h) for the full API and [`HOWTO.md`](HOWTO.md) for detailed usage.

## CLI

```bash
eakv inspect cache.eakv    # show metadata, sizes, compression ratio
eakv validate cache.eakv   # check for NaN/corruption via SIMD kernel
```

## Project Structure

```
include/eakv.h                Public API (opaque handle, C99)

src/
  cache.c                     Cache lifecycle, bulk quantization
  attention.c                 MHA/GQA dispatch (head_dim=64 and 128)
  io.c                        .eakv binary format save/load
  cli.c                       CLI tool

kernels/                      Ea SIMD kernels (.ea -> .o)
  quantize_simd.ea            Q4_1 quantization
  dequantize_{simd,avx2,avx512}.ea   Multi-ISA dequantization
  fused_k_score.ea            Fused Q@K^T (head_dim=128)
  fused_k_score_64.ea         Fused Q@K^T (head_dim=64)
  fused_v_sum.ea              Fused weights@V (head_dim=128)
  fused_v_sum_64.ea           Fused weights@V (head_dim=64)
  fused_k_score_gqa.ea        GQA K-score + V-sum (head_dim=128)
  fused_k_score_gqa_64.ea     GQA K-score + V-sum (head_dim=64)
  fused_attention.ea           Fused softmax+attention (2-pass)
  validate.ea                 NaN/negative scale detection

tests/                        C tests
benchmarks/                   C benchmarks
```

## Requirements

- x86-64 CPU with AVX-512
- GCC, Make
- [Ea compiler](https://github.com/niclas-edenworlds/eacompute) (for kernel compilation)
