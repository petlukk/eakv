# libeakv v0.2 Design

## Problem

eakv v0.1 is a Python library. Users must extract KV caches from their LLM runtime, convert to NumPy, quantize, save, load, and inject back. Nobody will do this. The technology works — Q4 quantization, fused SIMD attention, GQA — but the interface is wrong. eakv should be an internal engine inside an LLM runtime, not an external Python tool.

## Solution

Rewrite eakv as a C library (`libeakv.a` / `libeakv.so`) with a CLI tool for llama.cpp session file conversion. Kill the Python package.

## Target

Primary integration: llama.cpp. Two phases:

1. **Session converter** (v0.2) — CLI tool that converts llama.cpp session files to/from `.eakv`. Non-invasive, no fork needed. Users get 3x smaller session files and faster resume.
2. **KV backend replacement** (v0.3) — llama.cpp fork where eakv replaces the KV cache storage and attention path. Users get 3x less RAM during inference.

## C API (`eakv.h`)

```c
// Lifecycle
eakv_cache_t *eakv_cache_create(int n_layers, int n_kv_heads, int head_dim, int max_seq_len);
void          eakv_cache_free(eakv_cache_t *cache);

// Append — quantizes one token's KV on the fly
void eakv_cache_append(eakv_cache_t *cache, int layer,
                       const float *k_vec, const float *v_vec);

// Attention — operates directly on Q4 data
void eakv_attention_scores(const eakv_cache_t *cache, const float *queries,
                           int layer, int n_q_heads, int n_kv_heads,
                           float *scores_out);

void eakv_attention_output(const eakv_cache_t *cache, const float *weights,
                           int layer, int n_q_heads, int n_kv_heads,
                           float *output_out);

// File I/O
int  eakv_cache_save(const eakv_cache_t *cache, const char *path);
int  eakv_cache_load(const char *path, eakv_cache_t **out);

// llama.cpp session conversion
int  eakv_import_llama_session(const char *session_path, eakv_cache_t **out);
int  eakv_export_llama_session(const eakv_cache_t *cache, const char *session_path);

// Info
int   eakv_cache_seq_len(const eakv_cache_t *cache);
int   eakv_cache_n_layers(const eakv_cache_t *cache);
float eakv_cache_compression_ratio(const eakv_cache_t *cache);
```

## CLI

```bash
eakv import session.bin -o session.eakv    # llama.cpp -> eakv (3x smaller)
eakv export session.eakv -o session.bin    # eakv -> llama.cpp
eakv inspect session.eakv                  # show metadata
eakv validate session.eakv                 # check integrity
```

## Internal Architecture

```
eakv.h                    <- public API (one header)
src/
  cache.c                 <- eakv_cache_t struct, create/free/append/grow
  attention.c             <- dispatch to fused kernels (GQA/MHA)
  io.c                    <- .eakv binary format read/write
  llama_session.c         <- llama.cpp session import/export
  cli.c                   <- main() for eakv CLI tool
kernels/                  <- existing .ea files (unchanged)
  quantize_simd.ea
  dequantize_*.ea
  fused_k_score*.ea
  fused_v_sum*.ea
  validate.ea
build_kernels.sh          <- compiles .ea -> .o files
Makefile                  <- links kernel .o + C sources -> libeakv.a + eakv
```

## How Append Works

`eakv_cache_append()` receives one token's K and V vectors (f32, length = `n_kv_heads * head_dim`). It quantizes to Q4 in-place into the cache buffer. When a group of 64 values is complete, the SIMD quantize kernel runs. Until then, values accumulate in a small staging buffer.

The cache pre-allocates for `max_seq_len` tokens up front. No realloc during inference.

## Memory Layout of `eakv_cache_t`

```
Per layer, per K/V:
  weights: uint8[max_groups * 32]     (packed Q4 nibbles)
  scales:  float[max_groups]
  biases:  float[max_groups]
```

Flat allocation, one `malloc` per cache. Layers contiguous. Same layout as `.eakv` file format — `save()` is essentially `memcpy` + header.

## What Stays from v0.1

- All 9 Ea kernel files — unchanged
- `.eakv` file format — unchanged (64-byte aligned, mmap-ready)
- Q4_1 quantization scheme (group size 64) — unchanged

## What Dies

- All Python code (`src/eakv/*.py`)
- `pyproject.toml`, `pip install eakv`
- Python tests, Python benchmarks

## Testing

C tests using assert macros (no framework):
- Quantize -> dequantize roundtrip accuracy
- Fused attention correctness (MHA, GQA)
- File save -> load -> verify
- Append token by token -> verify same result as bulk quantize
- llama.cpp session round-trip (import -> export -> binary compare)

## Build

```bash
./build_kernels.sh          # .ea -> .o (needs ea compiler)
make                        # .o + .c -> libeakv.a + libeakv.so + eakv
make test                   # run C tests
```

## Decisions

- **Monolithic library** — one `libeakv.a`, one `eakv.h`. No layered split. YAGNI.
- **Kill Python** — not freeze, not wrap. Delete. C library is the product.
- **Pre-allocate** — `max_seq_len` set at create time. No realloc during inference.
- **llama.cpp first** — session converter is the proof-of-concept. KV backend replacement is v0.3.
- **Ea kernels unchanged** — the hard part (SIMD Q4 attention) is done. This is systems integration work.
