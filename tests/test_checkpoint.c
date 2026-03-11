#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include "eakv.h"

static void test_checkpoint_returns_seq_len(void) {
    eakv_cache_t *cache = eakv_cache_create(2, 2, 64, 256);
    assert(cache != NULL);
    int cp0 = eakv_checkpoint(cache);
    assert(cp0 == 0);
    int n_embd = 2 * 64;
    int seq = 8;
    float *data = calloc(2 * 2 * n_embd * seq, sizeof(float));
    eakv_cache_load_raw(cache, data, seq);
    free(data);
    int cp1 = eakv_checkpoint(cache);
    assert(cp1 == 8);
    printf("  PASS: checkpoint_returns_seq_len\n");
    eakv_cache_free(cache);
}

static void test_restore_resets_seq_len(void) {
    eakv_cache_t *cache = eakv_cache_create(2, 2, 64, 256);
    int n_embd = 2 * 64;
    float *data = calloc(2 * 2 * n_embd * 16, sizeof(float));
    eakv_cache_load_raw(cache, data, 16);
    free(data);
    int cp = eakv_checkpoint(cache);
    assert(cp == 16);
    eakv_restore(cache, 8);
    assert(eakv_cache_seq_len(cache) == 8);
    eakv_restore(cache, 0);
    assert(eakv_cache_seq_len(cache) == 0);
    printf("  PASS: restore_resets_seq_len\n");
    eakv_cache_free(cache);
}

static void test_restore_bounds_check(void) {
    eakv_cache_t *cache = eakv_cache_create(2, 2, 64, 256);
    int n_embd = 2 * 64;
    float *data = calloc(2 * 2 * n_embd * 8, sizeof(float));
    eakv_cache_load_raw(cache, data, 8);
    free(data);
    int rc = eakv_restore(cache, 16);
    assert(rc == EAKV_ERR_INVALID);
    assert(eakv_cache_seq_len(cache) == 8);
    rc = eakv_restore(cache, -1);
    assert(rc == EAKV_ERR_INVALID);
    assert(eakv_cache_seq_len(cache) == 8);
    printf("  PASS: restore_bounds_check\n");
    eakv_cache_free(cache);
}

static void test_append_after_restore(void) {
    eakv_cache_t *cache = eakv_cache_create(2, 2, 64, 256);
    int n_embd = 2 * 64;
    float *data = calloc(2 * 2 * n_embd * 16, sizeof(float));
    eakv_cache_load_raw(cache, data, 16);
    eakv_restore(cache, 8);
    float *append_data = calloc(n_embd * 4, sizeof(float));
    for (int l = 0; l < 2; l++) {
        eakv_cache_append(cache, append_data, l, 0, 4);
        eakv_cache_append(cache, append_data, l, 1, 4);
    }
    eakv_cache_advance(cache, 4);
    assert(eakv_cache_seq_len(cache) == 12);
    free(append_data);
    free(data);
    printf("  PASS: append_after_restore\n");
    eakv_cache_free(cache);
}

int main(void) {
    printf("test_checkpoint:\n");
    test_checkpoint_returns_seq_len();
    test_restore_resets_seq_len();
    test_restore_bounds_check();
    test_append_after_restore();
    printf("All checkpoint tests passed.\n");
    return 0;
}
