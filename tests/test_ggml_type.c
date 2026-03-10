/* Test eakv ggml type adapter: quantize, dequantize, vec_dot.
 * Verifies the AoS block format produces correct results.
 */
#include "eakv_ggml.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

static int test_roundtrip(void) {
    printf("TEST 1: quantize → dequantize roundtrip ...\n");

    const int n = 256;  /* 4 blocks */
    float *src = malloc(n * sizeof(float));
    float *dst = malloc(n * sizeof(float));
    block_q4_1_eakv *blocks = malloc((n / QK_EAKV) * sizeof(block_q4_1_eakv));

    /* Fill with known pattern */
    for (int i = 0; i < n; i++)
        src[i] = sinf(i * 0.1f) * 10.0f;

    eakv_quantize_row(src, blocks, n);
    eakv_dequantize_row(blocks, dst, n);

    /* Q4_1 with 16 levels over range → max error ≈ range/30 */
    double max_err = 0;
    for (int i = 0; i < n; i++) {
        double e = fabs(dst[i] - src[i]);
        if (e > max_err) max_err = e;
    }

    printf("  max roundtrip error: %.4f\n", max_err);
    int ok = max_err < 2.0;  /* generous for Q4 */
    printf("  %s\n\n", ok ? "PASSED" : "FAILED");

    free(src); free(dst); free(blocks);
    return ok;
}

static int test_vec_dot(void) {
    printf("TEST 2: vec_dot Q4×F32 vs reference ...\n");

    const int n = 128;  /* 2 blocks = head_dim=128 */
    float *kv = malloc(n * sizeof(float));
    float *query = malloc(n * sizeof(float));
    block_q4_1_eakv *blocks = malloc((n / QK_EAKV) * sizeof(block_q4_1_eakv));

    /* KV data: varied range */
    for (int i = 0; i < n; i++)
        kv[i] = cosf(i * 0.07f) * 5.0f + 2.0f;

    /* Query data */
    for (int i = 0; i < n; i++)
        query[i] = sinf(i * 0.13f) * 0.5f;

    /* Quantize KV */
    eakv_quantize_row(kv, blocks, n);

    /* vec_dot result */
    float dot_q4 = 0;
    eakv_vec_dot_q4_f32(n, &dot_q4, 0, blocks, 0, query, 0, 1);

    /* Dequantize and compute reference dot */
    float *kv_deq = malloc(n * sizeof(float));
    eakv_dequantize_row(blocks, kv_deq, n);

    float dot_ref = 0;
    for (int i = 0; i < n; i++)
        dot_ref += kv_deq[i] * query[i];

    /* vec_dot should exactly match dequant+dot (same data path) */
    double err = fabs(dot_q4 - dot_ref);
    printf("  vec_dot:  %.6f\n", dot_q4);
    printf("  ref dot:  %.6f\n", dot_ref);
    printf("  error:    %.6f\n", err);

    int ok = err < 0.01;  /* should be nearly identical */
    printf("  %s\n\n", ok ? "PASSED" : "FAILED");

    free(kv); free(query); free(blocks); free(kv_deq);
    return ok;
}

static int test_vec_dot_f32_ref(void) {
    printf("TEST 3: vec_dot Q4×F32 vs pure F32 ...\n");

    const int n = 64;  /* 1 block = head_dim=64 */
    float *kv = malloc(n * sizeof(float));
    float *query = malloc(n * sizeof(float));
    block_q4_1_eakv *blocks = malloc(sizeof(block_q4_1_eakv));

    for (int i = 0; i < n; i++) {
        kv[i] = (float)(i % 17) - 8.0f;
        query[i] = (float)(i % 11) / 10.0f - 0.5f;
    }

    eakv_quantize_row(kv, blocks, n);

    float dot_q4 = 0;
    eakv_vec_dot_q4_f32(n, &dot_q4, 0, blocks, 0, query, 0, 1);

    /* Pure f32 reference */
    float dot_f32 = 0;
    for (int i = 0; i < n; i++)
        dot_f32 += kv[i] * query[i];

    double err = fabs(dot_q4 - dot_f32);
    double rel = fabs(dot_f32) > 0.01 ? err / fabs(dot_f32) : err;
    printf("  Q4 dot:   %.6f\n", dot_q4);
    printf("  F32 dot:  %.6f\n", dot_f32);
    printf("  abs err:  %.4f\n", err);
    printf("  rel err:  %.2f%%\n", rel * 100);

    int ok = rel < 0.20;  /* Q4 quantization error */
    printf("  %s\n\n", ok ? "PASSED" : "FAILED");

    free(kv); free(query); free(blocks);
    return ok;
}

static int test_block_layout(void) {
    printf("TEST 4: block layout correctness ...\n");

    float src[64];
    for (int i = 0; i < 64; i++)
        src[i] = (float)i;  /* 0..63, range=63, scale=63/15=4.2 */

    block_q4_1_eakv block;
    eakv_quantize_row(src, &block, 64);

    printf("  scale=%.4f bias=%.4f\n", block.d, block.m);

    /* Dequantize and check ordering */
    float dst[64];
    eakv_dequantize_row(&block, dst, 64);

    /* Values should be monotonically increasing (approximately) */
    int monotonic = 1;
    for (int i = 1; i < 64; i++) {
        if (dst[i] < dst[i - 1] - 0.01f) {
            printf("  non-monotonic at %d: %.2f < %.2f\n", i, dst[i], dst[i-1]);
            monotonic = 0;
        }
    }

    printf("  monotonic: %s\n", monotonic ? "yes" : "no");
    printf("  first: %.2f last: %.2f (expect ~0, ~63)\n", dst[0], dst[63]);

    int ok = monotonic && dst[0] < 1.0f && dst[63] > 60.0f;
    printf("  %s\n\n", ok ? "PASSED" : "FAILED");
    return ok;
}

static int test_multiple_blocks(void) {
    printf("TEST 5: multi-block vec_dot (512 elements) ...\n");

    const int n = 512;
    float *kv = malloc(n * sizeof(float));
    float *query = malloc(n * sizeof(float));
    int nb = n / QK_EAKV;
    block_q4_1_eakv *blocks = malloc(nb * sizeof(block_q4_1_eakv));

    for (int i = 0; i < n; i++) {
        kv[i] = sinf(i * 0.05f) * 3.0f;
        query[i] = cosf(i * 0.03f) * 0.5f;
    }

    eakv_quantize_row(kv, blocks, n);

    float dot_q4 = 0;
    eakv_vec_dot_q4_f32(n, &dot_q4, 0, blocks, 0, query, 0, 1);

    /* Dequant reference */
    float *deq = malloc(n * sizeof(float));
    eakv_dequantize_row(blocks, deq, n);
    float dot_ref = 0;
    for (int i = 0; i < n; i++)
        dot_ref += deq[i] * query[i];

    double err = fabs(dot_q4 - dot_ref);
    printf("  Q4 dot:  %.6f\n", dot_q4);
    printf("  ref dot: %.6f\n", dot_ref);
    printf("  error:   %.6f\n", err);

    int ok = err < 0.1;
    printf("  %s\n\n", ok ? "PASSED" : "FAILED");

    free(kv); free(query); free(blocks); free(deq);
    return ok;
}

int main(void) {
    printf("=== test_ggml_type: eakv Q4_1 block format ===\n\n");

    int pass = 0, total = 0;

    total++; pass += test_roundtrip();
    total++; pass += test_vec_dot();
    total++; pass += test_vec_dot_f32_ref();
    total++; pass += test_block_layout();
    total++; pass += test_multiple_blocks();

    printf("=== %d/%d passed ===\n", pass, total);
    return pass == total ? 0 : 1;
}
