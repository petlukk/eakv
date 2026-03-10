#include "test_harness.h"
#include "eakv.h"
#include <string.h>
#include <unistd.h>

TEST(test_save_load_roundtrip) {
    int nl = 2, nh = 4, hd = 128, sl = 32;
    eakv_cache_t *cache = eakv_cache_create(nl, nh, hd, sl);

    int total = nl * 2 * nh * sl * hd;
    float *data = malloc(total * sizeof(float));
    for (int i = 0; i < total; i++)
        data[i] = (float)(i % 1000) / 100.0f - 5.0f;
    eakv_cache_load_raw(cache, data, sl);

    const char *path = "/tmp/test_libeakv_roundtrip.eakv";
    int rc = eakv_cache_save(cache, path);
    ASSERT_EQ(rc, EAKV_OK);

    eakv_cache_t *loaded = NULL;
    rc = eakv_cache_load(path, &loaded);
    ASSERT_EQ(rc, EAKV_OK);
    ASSERT(loaded != NULL);
    ASSERT_EQ(eakv_cache_seq_len(loaded), sl);
    ASSERT_EQ(eakv_cache_n_layers(loaded), nl);
    ASSERT_EQ(eakv_cache_n_heads(loaded), nh);
    ASSERT_EQ(eakv_cache_head_dim(loaded), hd);

    /* Verify attention produces same results from both caches */
    float q[4 * 128];
    for (int i = 0; i < 4 * 128; i++) q[i] = 1.0f;

    float scores1[4 * 32], scores2[4 * 32];
    eakv_attention_scores(cache, q, 0, 4, 4, scores1);
    eakv_attention_scores(loaded, q, 0, 4, 4, scores2);

    for (int i = 0; i < 4 * 32; i++)
        ASSERT_NEAR(scores1[i], scores2[i], 1e-6);

    free(data);
    eakv_cache_free(cache);
    eakv_cache_free(loaded);
    unlink(path);
}

TEST(test_load_nonexistent) {
    eakv_cache_t *cache = NULL;
    int rc = eakv_cache_load("/tmp/does_not_exist_libeakv.eakv", &cache);
    ASSERT_EQ(rc, EAKV_ERR_IO);
    ASSERT(cache == NULL);
}

TEST(test_load_bad_magic) {
    const char *path = "/tmp/test_bad_magic.eakv";
    FILE *f = fopen(path, "wb");
    char bad[512] = {0};
    memcpy(bad, "NOPE", 4);
    fwrite(bad, 1, 512, f);
    fclose(f);

    eakv_cache_t *cache = NULL;
    int rc = eakv_cache_load(path, &cache);
    ASSERT_EQ(rc, EAKV_ERR_FORMAT);
    unlink(path);
}

TEST_MAIN()
    RUN(test_save_load_roundtrip);
    RUN(test_load_nonexistent);
    RUN(test_load_bad_magic);
TEST_END()
