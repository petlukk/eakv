#include "internal.h"
#include <stdlib.h>
#include <string.h>

static size_t block_size(int max_groups) {
    return (size_t)max_groups * 32
         + (size_t)max_groups * sizeof(float)
         + (size_t)max_groups * sizeof(float);
}

eakv_cache_t *eakv_cache_create(int n_layers, int n_kv_heads,
                                 int head_dim, int max_seq_len) {
    if (n_layers <= 0 || n_kv_heads <= 0 || head_dim <= 0 || max_seq_len <= 0)
        return NULL;
    if ((n_kv_heads * head_dim) % 64 != 0)
        return NULL;

    eakv_cache_t *c = calloc(1, sizeof(*c));
    if (!c) return NULL;

    c->n_layers    = n_layers;
    c->n_kv_heads  = n_kv_heads;
    c->head_dim    = head_dim;
    c->max_seq_len = max_seq_len;
    c->seq_len     = 0;
    c->groups_per_token = (n_kv_heads * head_dim) / 64;
    c->max_groups       = c->groups_per_token * max_seq_len;

    c->kv = calloc(n_layers * 2, sizeof(eakv_kv_data_t));
    if (!c->kv) { free(c); return NULL; }

    size_t blk = block_size(c->max_groups);
    size_t total = blk * n_layers * 2;
    c->data_buf = calloc(1, total);
    if (!c->data_buf) { free(c->kv); free(c); return NULL; }

    char *ptr = (char *)c->data_buf;
    for (int l = 0; l < n_layers; l++) {
        for (int kv = 0; kv < 2; kv++) {
            eakv_kv_data_t *d = &c->kv[l * 2 + kv];
            d->weights = (uint8_t *)ptr;
            ptr += (size_t)c->max_groups * 32;
            d->scales = (float *)ptr;
            ptr += (size_t)c->max_groups * sizeof(float);
            d->biases = (float *)ptr;
            ptr += (size_t)c->max_groups * sizeof(float);
        }
    }

    return c;
}

void eakv_cache_free(eakv_cache_t *cache) {
    if (!cache) return;
    free(cache->data_buf);
    free(cache->kv);
    free(cache);
}

int eakv_cache_load_raw(eakv_cache_t *cache, const float *data, int seq_len) {
    if (!cache || !data || seq_len <= 0)
        return EAKV_ERR_INVALID;
    if (seq_len > cache->max_seq_len)
        return EAKV_ERR_INVALID;

    int elems_per_lkv = cache->n_kv_heads * seq_len * cache->head_dim;
    int n_groups = elems_per_lkv / 64;

    int32_t *tmp = malloc(n_groups * 32 * sizeof(int32_t));
    if (!tmp) return EAKV_ERR_ALLOC;

    for (int l = 0; l < cache->n_layers; l++) {
        for (int kv = 0; kv < 2; kv++) {
            const float *src = data + (l * 2 + kv) * elems_per_lkv;
            eakv_kv_data_t *d = &cache->kv[l * 2 + kv];

            q4_quantize_split_f32(src, tmp, d->scales, d->biases, n_groups);

            for (int i = 0; i < n_groups * 32; i++)
                d->weights[i] = (uint8_t)tmp[i];
        }
    }

    cache->seq_len = seq_len;
    free(tmp);
    return EAKV_OK;
}

int eakv_cache_seq_len(const eakv_cache_t *cache) {
    return cache ? cache->seq_len : 0;
}

int eakv_cache_n_layers(const eakv_cache_t *cache) {
    return cache ? cache->n_layers : 0;
}

int eakv_cache_n_heads(const eakv_cache_t *cache) {
    return cache ? cache->n_kv_heads : 0;
}

int eakv_cache_head_dim(const eakv_cache_t *cache) {
    return cache ? cache->head_dim : 0;
}

int eakv_cache_max_seq_len(const eakv_cache_t *cache) {
    return cache ? cache->max_seq_len : 0;
}

float eakv_cache_compression_ratio(const eakv_cache_t *cache) {
    if (!cache || cache->seq_len == 0) return 0.0f;
    return 40.0f / 256.0f;
}
