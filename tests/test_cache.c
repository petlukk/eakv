#include "test_harness.h"
#include "eakv.h"

TEST(test_create_and_free) {
    eakv_cache_t *cache = eakv_cache_create(2, 8, 128, 64);
    ASSERT(cache != NULL);
    ASSERT_EQ(eakv_cache_seq_len(cache), 0);
    ASSERT_EQ(eakv_cache_n_layers(cache), 2);
    ASSERT_EQ(eakv_cache_n_heads(cache), 8);
    ASSERT_EQ(eakv_cache_head_dim(cache), 128);
    ASSERT_EQ(eakv_cache_max_seq_len(cache), 64);
    eakv_cache_free(cache);
}

TEST(test_create_rejects_bad_params) {
    ASSERT(eakv_cache_create(0, 8, 128, 64) == NULL);
    ASSERT(eakv_cache_create(2, 0, 128, 64) == NULL);
    ASSERT(eakv_cache_create(2, 8, 128, 0)  == NULL);
    ASSERT(eakv_cache_create(2, 3, 100, 64) == NULL);  /* 300 not % 64 */
}

TEST(test_load_raw) {
    int nl = 1, nh = 2, hd = 128, sl = 64;
    eakv_cache_t *cache = eakv_cache_create(nl, nh, hd, sl);
    ASSERT(cache != NULL);

    int total = nl * 2 * nh * sl * hd;
    float *data = malloc(total * sizeof(float));
    for (int i = 0; i < total; i++)
        data[i] = (float)(i % 1000) / 100.0f - 5.0f;

    int rc = eakv_cache_load_raw(cache, data, sl);
    ASSERT_EQ(rc, EAKV_OK);
    ASSERT_EQ(eakv_cache_seq_len(cache), sl);

    free(data);
    eakv_cache_free(cache);
}

TEST(test_load_raw_rejects_oversized) {
    eakv_cache_t *cache = eakv_cache_create(1, 2, 128, 32);
    float dummy = 0.0f;
    ASSERT_EQ(eakv_cache_load_raw(cache, &dummy, 64), EAKV_ERR_INVALID);
    eakv_cache_free(cache);
}

TEST(test_free_null_is_safe) {
    eakv_cache_free(NULL);
}

TEST_MAIN()
    RUN(test_create_and_free);
    RUN(test_create_rejects_bad_params);
    RUN(test_load_raw);
    RUN(test_load_raw_rejects_oversized);
    RUN(test_free_null_is_safe);
TEST_END()
