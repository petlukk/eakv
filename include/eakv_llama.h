/* eakv_llama.h — Optional llama.cpp bridge.
 *
 * Parses llama_state_seq_get_data() output into an eakv cache.
 * Requires llama.cpp headers (llama.h, ggml.h) for F16 conversion.
 */
#ifndef EAKV_LLAMA_H
#define EAKV_LLAMA_H

#include "eakv.h"
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Parse a llama.cpp sequence state buffer and create a populated eakv cache.
 *
 * state_buf/state_size: output from llama_state_seq_get_data()
 * n_layers, n_kv_heads, head_dim: model architecture params
 * max_seq_len: maximum context length for the cache
 * out: receives the new cache (caller must eakv_cache_free)
 *
 * Handles F16->f32 conversion and [pos][head][dim] -> [head][pos][dim] transpose.
 */
int eakv_from_llama_state(const uint8_t *state_buf, size_t state_size,
                          int n_layers, int n_kv_heads, int head_dim,
                          int max_seq_len, eakv_cache_t **out);

#ifdef __cplusplus
}
#endif

#endif /* EAKV_LLAMA_H */
