#include "test_harness.h"
#include "eakv.h"
#include <string.h>
#include <unistd.h>

TEST(test_full_pipeline) {
    int nl = 4, nh = 8, hd = 128, sl = 256;
    eakv_cache_t *cache = eakv_cache_create(nl, nh, hd, sl);
    ASSERT(cache != NULL);

    int total = nl * 2 * nh * sl * hd;
    float *data = malloc(total * sizeof(float));
    for (int i = 0; i < total; i++)
        data[i] = (float)(i % 2000) / 200.0f - 5.0f;

    ASSERT_EQ(eakv_cache_load_raw(cache, data, sl), EAKV_OK);

    float q[8 * 128];
    for (int i = 0; i < 8 * 128; i++) q[i] = 0.1f;

    float scores[8 * 256];
    float w[8 * 256];
    float out[8 * 128];

    for (int l = 0; l < nl; l++) {
        eakv_attention_scores(cache, q, l, 8, 8, scores);
        for (int i = 0; i < 8 * 256; i++)
            ASSERT(scores[i] == scores[i]);

        for (int i = 0; i < 8 * 256; i++) w[i] = 1.0f / 256.0f;

        memset(out, 0, sizeof(out));
        eakv_attention_output(cache, w, l, 8, 8, out);
        for (int i = 0; i < 8 * 128; i++)
            ASSERT(out[i] == out[i]);
    }

    const char *path = "/tmp/test_libeakv_e2e.eakv";
    ASSERT_EQ(eakv_cache_save(cache, path), EAKV_OK);

    eakv_cache_t *loaded = NULL;
    ASSERT_EQ(eakv_cache_load(path, &loaded), EAKV_OK);
    ASSERT_EQ(eakv_cache_seq_len(loaded), sl);
    ASSERT_EQ(eakv_cache_n_layers(loaded), nl);

    float scores2[8 * 256];
    eakv_attention_scores(cache, q, 0, 8, 8, scores);
    eakv_attention_scores(loaded, q, 0, 8, 8, scores2);

    for (int i = 0; i < 8 * 256; i++)
        ASSERT_NEAR(scores[i], scores2[i], 1e-6);

    free(data);
    eakv_cache_free(cache);
    eakv_cache_free(loaded);
    unlink(path);
}

TEST_MAIN()
    RUN(test_full_pipeline);
TEST_END()
