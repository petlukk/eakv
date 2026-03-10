/* libeakv — Q4 KV cache quantization for LLM inference.
 *
 * Single-header public API. All functions are thread-safe for distinct caches.
 */
#ifndef EAKV_H
#define EAKV_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Opaque cache handle */
typedef struct eakv_cache eakv_cache_t;

/* Error codes */
#define EAKV_OK              0
#define EAKV_ERR_IO         -1
#define EAKV_ERR_FORMAT     -2
#define EAKV_ERR_ALLOC      -3
#define EAKV_ERR_INVALID    -4

/* Lifecycle */
eakv_cache_t *eakv_cache_create(int n_layers, int n_kv_heads,
                                 int head_dim, int max_seq_len);
void          eakv_cache_free(eakv_cache_t *cache);

/* Bulk load — quantize a full f32 KV cache at once.
 * data: f32[n_layers * 2 * n_kv_heads * seq_len * head_dim]
 * Layout: [layer][kv][head][pos][dim] */
int eakv_cache_load_raw(eakv_cache_t *cache, const float *data, int seq_len);

/* Attention — operates directly on Q4 data.
 *
 * MHA: n_q_heads == n_kv_heads
 * GQA: n_q_heads > n_kv_heads (multiple Q heads share each KV head)
 *
 * queries:     f32[n_q_heads * head_dim]
 * scores_out:  f32[n_q_heads * seq_len]
 * weights:     f32[n_q_heads * seq_len]  (softmax weights)
 * output_out:  f32[n_q_heads * head_dim]
 */
void eakv_attention_scores(const eakv_cache_t *cache, const float *queries,
                           int layer, int n_q_heads, int n_kv_heads,
                           float *scores_out);

void eakv_attention_output(const eakv_cache_t *cache, const float *weights,
                           int layer, int n_q_heads, int n_kv_heads,
                           float *output_out);

/* File I/O — .eakv binary format */
int eakv_cache_save(const eakv_cache_t *cache, const char *path);
int eakv_cache_load(const char *path, eakv_cache_t **out);

/* Info */
int   eakv_cache_seq_len(const eakv_cache_t *cache);
int   eakv_cache_n_layers(const eakv_cache_t *cache);
int   eakv_cache_n_heads(const eakv_cache_t *cache);
int   eakv_cache_head_dim(const eakv_cache_t *cache);
int   eakv_cache_max_seq_len(const eakv_cache_t *cache);
float eakv_cache_compression_ratio(const eakv_cache_t *cache);

#ifdef __cplusplus
}
#endif

#endif /* EAKV_H */
