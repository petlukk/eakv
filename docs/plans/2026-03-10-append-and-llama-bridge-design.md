# Design: Incremental Append + llama.cpp Integration Layer

## Overview

Add incremental token append to eakv (currently only bulk `load_raw`) and a
llama.cpp bridge that parses state buffers into eakv caches. Core API stays
engine-agnostic; llama.cpp coupling is isolated in a separate optional module.

## 1. Core API: `eakv_cache_append`

```c
int eakv_cache_append(eakv_cache_t *cache, const float *data,
                      int layer, int kv_idx, int n_tokens);
```

- `data`: f32[n_kv_heads * n_tokens * head_dim], layout [head][token][dim]
- `kv_idx`: 0=K, 1=V
- Quantizes only the new tokens, writes at `seq_len` offset
- Increments `cache->seq_len` by `n_tokens` after successful append
- Returns `EAKV_ERR_INVALID` if `seq_len + n_tokens > max_seq_len`

### Implementation

The quantizer operates on groups of 64. Each token produces
`n_kv_heads * head_dim / 64` groups. For append:

1. Compute `group_offset = groups_per_token * cache->seq_len`
2. Compute `n_new_groups = groups_per_token * n_tokens`
3. Call `q4_quantize_split_f32` on the new data
4. Pack int32 weights to uint8 at the correct offset in the kv buffer
5. Only increment `seq_len` after all layers+kv have been appended

Note: seq_len must only increment once per full token append (all layers, K+V).
The caller is responsible for calling append for each layer and kv_idx before
the token is considered committed. A helper `eakv_cache_commit` could formalize
this, but for now the caller increments by calling append with the last kv_idx.

Actually, simpler: `eakv_cache_append` writes at the offset but does NOT
increment seq_len. A separate `eakv_cache_set_seq_len` or the append itself
increments. Since layers are independent in the buffer, each append can write
independently. The seq_len just needs to reflect the committed length before
attention is called.

Decision: append writes at `cache->seq_len` offset and increments seq_len
by n_tokens. The caller must append all layers and both K/V before calling
attention. This matches how inference engines work — they produce all KV
for a token before running attention on the next token.

## 2. Core API: `eakv_cache_clear`

```c
void eakv_cache_clear(eakv_cache_t *cache);
```

Resets `seq_len` to 0. No deallocation. Cache can be reused for a new sequence.

## 3. llama.cpp Bridge: `eakv_llama.h`

```c
int eakv_from_llama_state(const uint8_t *state_buf, size_t state_size,
                          int n_layers, int n_kv_heads, int head_dim,
                          int max_seq_len, eakv_cache_t **out);
```

- Parses `llama_state_seq_get_data()` binary format
- Handles F16→f32 conversion (calls `ggml_fp16_to_fp32`)
- Transposes [pos][head][dim] → [head][pos][dim]
- Creates cache via `eakv_cache_create`, populates via `eakv_cache_append`
- Returns error codes on parse failure

### Binary format parsed

```
[n_stream: u32][cell_count: u32]
[cell metadata: pos(i32) + n_seq_id(u32) + seq_ids(i32*n)]...
[v_trans: u32][n_layer: u32]
For each layer K:
  [type: i32][size_row: u64][data: f16[cell_count * n_embd_k_gqa]]
For each layer V:
  [type: i32][size_row: u64][data: f16[...]]
```

## 4. File Layout

- `include/eakv.h` — add `eakv_cache_append`, `eakv_cache_clear`
- `src/cache.c` — implement append and clear
- `include/eakv_llama.h` — llama bridge header (optional)
- `src/llama_bridge.c` — llama state parser (optional, needs llama.cpp headers)
- `tests/test_append.c` — verify append matches bulk load_raw
- `tests/test_llama_bridge.c` — verify bridge against test_llama_kv.c results

## 5. Testing Strategy

`test_append.c`: Create cache, bulk-load with `load_raw`, create second cache,
append token-by-token, run attention on both, verify scores match exactly
(bit-identical, since same quantization path).

`test_llama_bridge.c`: Use `eakv_from_llama_state` to load TinyLlama state,
verify quantization quality and attention accuracy match the known-good results
from the benchmark doc (19.1 dB SNR, 15.45% mean relative error).
