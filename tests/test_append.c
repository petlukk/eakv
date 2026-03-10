#include "eakv.h"
#include "internal.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

static int tests_run = 0, tests_failed = 0;

#define CHECK(name, cond) do { \
    tests_run++; \
    if (!(cond)) { printf("  FAIL: %s\n", name); tests_failed++; } \
    else { printf("  ok:   %s\n", name); } \
} while(0)

/* Generate deterministic test data: [n_layers][2][n_heads][seq_len][head_dim] */
static float *make_data(int nl, int nh, int hd, int sl) {
    int total = nl * 2 * nh * sl * hd;
    float *d = malloc((size_t)total * sizeof(float));
    for (int i = 0; i < total; i++)
        d[i] = (float)(i % 2000) / 200.0f - 5.0f;
    return d;
}

/* Extract per-layer/kv slice in [head][token][dim] layout from bulk data */
static float *extract_slice(const float *bulk, int nl, int nh, int hd, int sl,
                            int layer, int kv_idx, int tok_start, int n_tok) {
    float *out = malloc((size_t)nh * n_tok * hd * sizeof(float));
    int elems_per_lkv = nh * sl * hd;
    const float *base = bulk + (layer * 2 + kv_idx) * elems_per_lkv;
    /* bulk layout: [head][pos][dim] — extract [head][tok_start..][dim] */
    for (int h = 0; h < nh; h++)
        for (int t = 0; t < n_tok; t++)
            memcpy(out + (h * n_tok + t) * hd,
                   base + (h * sl + tok_start + t) * hd,
                   hd * sizeof(float));
    return out;
}

static void test_append_matches_bulk(int nl, int nh, int hd, int sl,
                                     const char *label) {
    float *data = make_data(nl, nh, hd, sl);

    /* Bulk load */
    eakv_cache_t *bulk = eakv_cache_create(nl, nh, hd, sl);
    eakv_cache_load_raw(bulk, data, sl);

    /* Incremental append: one token at a time */
    eakv_cache_t *incr = eakv_cache_create(nl, nh, hd, sl);

    for (int t = 0; t < sl; t++) {
        for (int l = 0; l < nl; l++) {
            for (int kv = 0; kv < 2; kv++) {
                float *slice = extract_slice(data, nl, nh, hd, sl, l, kv, t, 1);
                int rc = eakv_cache_append(incr, slice, l, kv, 1);
                free(slice);
                if (rc != EAKV_OK) {
                    printf("  FAIL: %s — append returned %d at t=%d l=%d kv=%d\n",
                           label, rc, t, l, kv);
                    tests_run++; tests_failed++;
                    goto cleanup;
                }
            }
        }
        eakv_cache_advance(incr, 1);
    }

    /* Compare: run attention on both, scores must match */
    {
        int n_q = nh;
        float *queries = malloc((size_t)n_q * hd * sizeof(float));
        float *scores_bulk = malloc((size_t)n_q * sl * sizeof(float));
        float *scores_incr = malloc((size_t)n_q * sl * sizeof(float));
        for (int i = 0; i < n_q * hd; i++) queries[i] = 0.1f;

        eakv_attention_scores(bulk, queries, 0, n_q, nh, scores_bulk);
        eakv_attention_scores(incr, queries, 0, n_q, nh, scores_incr);

        float max_err = 0;
        for (int i = 0; i < n_q * sl; i++) {
            float e = fabsf(scores_bulk[i] - scores_incr[i]);
            if (e > max_err) max_err = e;
        }

        char buf[256];
        snprintf(buf, sizeof(buf), "%s — scores match (max_err=%.6f)", label, max_err);
        CHECK(buf, max_err < 1e-5f);

        free(queries); free(scores_bulk); free(scores_incr);
    }

    /* Also compare V output */
    {
        float *weights = malloc((size_t)nh * sl * sizeof(float));
        float *out_bulk = calloc(nh * hd, sizeof(float));
        float *out_incr = calloc(nh * hd, sizeof(float));
        for (int i = 0; i < nh * sl; i++) weights[i] = 1.0f / sl;

        eakv_attention_output(bulk, weights, 0, nh, nh, out_bulk);
        eakv_attention_output(incr, weights, 0, nh, nh, out_incr);

        float max_err = 0;
        for (int i = 0; i < nh * hd; i++) {
            float e = fabsf(out_bulk[i] - out_incr[i]);
            if (e > max_err) max_err = e;
        }

        char buf[256];
        snprintf(buf, sizeof(buf), "%s — V output match (max_err=%.6f)", label, max_err);
        CHECK(buf, max_err < 1e-5f);

        free(weights); free(out_bulk); free(out_incr);
    }

cleanup:
    free(data);
    eakv_cache_free(bulk);
    eakv_cache_free(incr);
}

static void test_batch_append(void) {
    int nl = 1, nh = 4, hd = 64, sl = 32;
    float *data = make_data(nl, nh, hd, sl);

    eakv_cache_t *bulk = eakv_cache_create(nl, nh, hd, sl);
    eakv_cache_load_raw(bulk, data, sl);

    /* Append in batches of 8 */
    eakv_cache_t *batch = eakv_cache_create(nl, nh, hd, sl);
    for (int t = 0; t < sl; t += 8) {
        int n = (t + 8 <= sl) ? 8 : sl - t;
        for (int l = 0; l < nl; l++) {
            for (int kv = 0; kv < 2; kv++) {
                float *slice = extract_slice(data, nl, nh, hd, sl, l, kv, t, n);
                eakv_cache_append(batch, slice, l, kv, n);
                free(slice);
            }
        }
        eakv_cache_advance(batch, n);
    }

    float *queries = malloc((size_t)nh * hd * sizeof(float));
    float *s1 = malloc((size_t)nh * sl * sizeof(float));
    float *s2 = malloc((size_t)nh * sl * sizeof(float));
    for (int i = 0; i < nh * hd; i++) queries[i] = 0.1f;

    eakv_attention_scores(bulk, queries, 0, nh, nh, s1);
    eakv_attention_scores(batch, queries, 0, nh, nh, s2);

    float max_err = 0;
    for (int i = 0; i < nh * sl; i++) {
        float e = fabsf(s1[i] - s2[i]);
        if (e > max_err) max_err = e;
    }
    CHECK("batch append (8 at a time) matches bulk", max_err < 1e-5f);

    free(data); free(queries); free(s1); free(s2);
    eakv_cache_free(bulk); eakv_cache_free(batch);
}

static void test_clear_and_reuse(void) {
    int nl = 1, nh = 4, hd = 64, sl = 16;
    eakv_cache_t *cache = eakv_cache_create(nl, nh, hd, sl);
    float *data = make_data(nl, nh, hd, sl);
    eakv_cache_load_raw(cache, data, sl);

    CHECK("seq_len before clear", eakv_cache_seq_len(cache) == sl);
    eakv_cache_clear(cache);
    CHECK("seq_len after clear", eakv_cache_seq_len(cache) == 0);

    /* Reuse: load again */
    eakv_cache_load_raw(cache, data, sl);
    CHECK("seq_len after reload", eakv_cache_seq_len(cache) == sl);

    free(data);
    eakv_cache_free(cache);
}

static void test_append_overflow(void) {
    int nl = 1, nh = 4, hd = 64, sl = 8;
    eakv_cache_t *cache = eakv_cache_create(nl, nh, hd, sl);
    float *data = make_data(nl, nh, hd, sl);
    eakv_cache_load_raw(cache, data, sl);

    /* Try to advance beyond max */
    int rc = eakv_cache_advance(cache, 1);
    CHECK("advance beyond max rejected", rc == EAKV_ERR_INVALID);

    /* Try to append beyond max */
    float tok[4 * 64];
    memset(tok, 0, sizeof(tok));
    rc = eakv_cache_append(cache, tok, 0, 0, 1);
    CHECK("append beyond max rejected", rc == EAKV_ERR_INVALID);

    free(data);
    eakv_cache_free(cache);
}

static void test_append_bad_params(void) {
    int nl = 2, nh = 4, hd = 64, sl = 16;
    eakv_cache_t *cache = eakv_cache_create(nl, nh, hd, sl);
    float tok[4 * 64];

    CHECK("append null cache", eakv_cache_append(NULL, tok, 0, 0, 1) == EAKV_ERR_INVALID);
    CHECK("append null data", eakv_cache_append(cache, NULL, 0, 0, 1) == EAKV_ERR_INVALID);
    CHECK("append bad layer", eakv_cache_append(cache, tok, 5, 0, 1) == EAKV_ERR_INVALID);
    CHECK("append bad kv_idx", eakv_cache_append(cache, tok, 0, 2, 1) == EAKV_ERR_INVALID);
    CHECK("append zero tokens", eakv_cache_append(cache, tok, 0, 0, 0) == EAKV_ERR_INVALID);

    eakv_cache_free(cache);
}

int main(void) {
    printf("\n=== test_append ===\n\n");

    test_append_matches_bulk(1, 4, 64, 32, "1L/4H/64d/32seq");
    test_append_matches_bulk(1, 4, 128, 16, "1L/4H/128d/16seq");
    test_append_matches_bulk(2, 8, 128, 8, "2L/8H/128d/8seq");
    test_batch_append();
    test_clear_and_reuse();
    test_append_overflow();
    test_append_bad_params();

    printf("\n%d tests, %d failed\n\n", tests_run, tests_failed);
    return tests_failed ? 1 : 0;
}
