#include "test_harness.h"
#include "eakv.h"
#include <string.h>

static void fill_linear(float *buf, int n) {
    for (int i = 0; i < n; i++)
        buf[i] = (float)(i % 256) / 50.0f - 2.5f;
}

TEST(test_mha_scores_finite) {
    int nl = 1, nh = 4, hd = 128, sl = 32;
    eakv_cache_t *cache = eakv_cache_create(nl, nh, hd, sl);

    int total = nl * 2 * nh * sl * hd;
    float *data = malloc(total * sizeof(float));
    fill_linear(data, total);
    eakv_cache_load_raw(cache, data, sl);

    float q[4 * 128];
    for (int i = 0; i < 4 * 128; i++) q[i] = 1.0f;

    float scores[4 * 32];
    eakv_attention_scores(cache, q, 0, 4, 4, scores);

    for (int i = 0; i < 4 * 32; i++)
        ASSERT(scores[i] == scores[i]);

    free(data);
    eakv_cache_free(cache);
}

TEST(test_mha_output_finite) {
    int nl = 1, nh = 4, hd = 128, sl = 32;
    eakv_cache_t *cache = eakv_cache_create(nl, nh, hd, sl);

    int total = nl * 2 * nh * sl * hd;
    float *data = malloc(total * sizeof(float));
    fill_linear(data, total);
    eakv_cache_load_raw(cache, data, sl);

    float w[4 * 32];
    for (int i = 0; i < 4 * 32; i++) w[i] = 1.0f / 32.0f;

    float out[4 * 128];
    memset(out, 0, sizeof(out));
    eakv_attention_output(cache, w, 0, 4, 4, out);

    for (int i = 0; i < 4 * 128; i++)
        ASSERT(out[i] == out[i]);

    free(data);
    eakv_cache_free(cache);
}

TEST(test_scores_deterministic) {
    int nl = 1, nh = 2, hd = 128, sl = 16;
    eakv_cache_t *c1 = eakv_cache_create(nl, nh, hd, sl);
    eakv_cache_t *c2 = eakv_cache_create(nl, nh, hd, sl);

    int total = nl * 2 * nh * sl * hd;
    float *data = malloc(total * sizeof(float));
    fill_linear(data, total);

    eakv_cache_load_raw(c1, data, sl);
    eakv_cache_load_raw(c2, data, sl);

    float q[2 * 128];
    for (int i = 0; i < 2 * 128; i++) q[i] = 0.5f;

    float s1[2 * 16], s2[2 * 16];
    eakv_attention_scores(c1, q, 0, 2, 2, s1);
    eakv_attention_scores(c2, q, 0, 2, 2, s2);

    for (int i = 0; i < 2 * 16; i++)
        ASSERT_NEAR(s1[i], s2[i], 1e-6);

    free(data);
    eakv_cache_free(c1);
    eakv_cache_free(c2);
}

TEST_MAIN()
    RUN(test_mha_scores_finite);
    RUN(test_mha_output_finite);
    RUN(test_scores_deterministic);
TEST_END()
