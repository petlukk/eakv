#define _POSIX_C_SOURCE 199309L
#include "eakv.h"
#include "internal.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include <unistd.h>

static double now_us(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e6 + ts.tv_nsec / 1e3;
}

static double bench(const char *name, void (*fn)(void *), void *ctx,
                    int warmup, int runs) {
    for (int i = 0; i < warmup; i++) fn(ctx);

    double times[100];
    if (runs > 100) runs = 100;
    for (int i = 0; i < runs; i++) {
        double t0 = now_us();
        fn(ctx);
        times[i] = now_us() - t0;
    }

    /* median */
    for (int i = 0; i < runs - 1; i++)
        for (int j = i + 1; j < runs; j++)
            if (times[j] < times[i]) {
                double t = times[i]; times[i] = times[j]; times[j] = t;
            }

    double med = times[runs / 2];
    printf("  %-40s %8.0f us (median of %d)\n", name, med, runs);
    return med;
}

typedef struct {
    eakv_cache_t *cache;
    float *queries;
    float *scores;
    float *weights;
    float *output;
    int n_heads;
    int seq_len;
    int head_dim;
} bench_ctx_t;

static void bench_k_scores(void *ctx) {
    bench_ctx_t *b = ctx;
    eakv_attention_scores(b->cache, b->queries, 0,
                          b->n_heads, b->n_heads, b->scores);
}

static void bench_v_output(void *ctx) {
    bench_ctx_t *b = ctx;
    memset(b->output, 0, (size_t)b->n_heads * b->head_dim * sizeof(float));
    eakv_attention_output(b->cache, b->weights, 0,
                          b->n_heads, b->n_heads, b->output);
}

/* F16 baseline: dequantize K to f32, then naive dot product */
typedef struct {
    eakv_cache_t *cache;
    float *queries;
    float *scores;
    float *f32_buf;  /* dequantized K data */
    int n_heads;
    int seq_len;
    int head_dim;
} baseline_ctx_t;

static void bench_baseline_k_scores(void *ctx) {
    baseline_ctx_t *b = ctx;
    eakv_kv_data_t *k = &b->cache->kv[0];  /* layer 0, K */
    int n_groups = b->cache->groups_per_token * b->seq_len;

    /* Step 1: dequantize Q4 -> f32 */
    q4_dequantize_avx512_f32(k->weights, k->scales, k->biases,
                              b->f32_buf, n_groups);

    /* Step 2: scaled dot product scores[h][t] = dot(query[h], K[h][t]) / sqrt(d) */
    int hd = b->head_dim;
    float scale = 1.0f / sqrtf((float)hd);
    for (int h = 0; h < b->n_heads; h++) {
        const float *q = b->queries + h * hd;
        for (int t = 0; t < b->seq_len; t++) {
            const float *kv = b->f32_buf + (h * b->seq_len + t) * hd;
            float sum = 0.0f;
            for (int d = 0; d < hd; d++)
                sum += q[d] * kv[d];
            b->scores[h * b->seq_len + t] = sum * scale;
        }
    }
}

/* Pure f32 baseline: no quantization at all, just dot product on raw f32 */
typedef struct {
    float *f32_data;  /* raw f32 K data */
    float *queries;
    float *scores;
    int n_heads;
    int seq_len;
    int head_dim;
} f32_ctx_t;

static void bench_f32_k_scores(void *ctx) {
    f32_ctx_t *b = ctx;
    int hd = b->head_dim;
    float scale = 1.0f / sqrtf((float)hd);
    for (int h = 0; h < b->n_heads; h++) {
        const float *q = b->queries + h * hd;
        for (int t = 0; t < b->seq_len; t++) {
            const float *kv = b->f32_data + (h * b->seq_len + t) * hd;
            float sum = 0.0f;
            for (int d = 0; d < hd; d++)
                sum += q[d] * kv[d];
            b->scores[h * b->seq_len + t] = sum * scale;
        }
    }
}

typedef struct {
    eakv_cache_t *cache;
    const char *path;
} io_ctx_t;

static void bench_save(void *ctx) {
    io_ctx_t *b = ctx;
    eakv_cache_save(b->cache, b->path);
}

static void bench_load(void *ctx) {
    io_ctx_t *b = ctx;
    eakv_cache_t *c = NULL;
    eakv_cache_load(b->path, &c);
    eakv_cache_free(c);
}

typedef struct {
    eakv_cache_t *cache;
    float *data;
    int seq_len;
} load_ctx_t;

static void bench_load_raw(void *ctx) {
    load_ctx_t *b = ctx;
    eakv_cache_load_raw(b->cache, b->data, b->seq_len);
}

int main(void) {
    struct { const char *name; int nl; int nh; int hd; int sl; } configs[] = {
        {"7B-like (1L, 8H, 2K seq)",  1,  8, 128, 2048},
        {"7B-like (1L, 8H, 8K seq)",  1,  8, 128, 8192},
        {"7B-like (32L, 8H, 2K seq)", 32, 8, 128, 2048},
    };
    int n_configs = (int)(sizeof(configs) / sizeof(configs[0]));

    for (int c = 0; c < n_configs; c++) {
        int nl = configs[c].nl, nh = configs[c].nh;
        int hd = configs[c].hd, sl = configs[c].sl;

        printf("\n%s\n", configs[c].name);
        printf("  (%d layers, %d heads, %d dim, %d seq)\n\n", nl, nh, hd, sl);

        int total = nl * 2 * nh * sl * hd;
        float *data = malloc((size_t)total * sizeof(float));
        for (int i = 0; i < total; i++)
            data[i] = (float)(i % 2000) / 200.0f - 5.0f;

        eakv_cache_t *cache = eakv_cache_create(nl, nh, hd, sl);

        load_ctx_t lctx = { cache, data, sl };
        bench("load_raw (quantize)", bench_load_raw, &lctx, 2, 10);

        float *queries = malloc((size_t)nh * hd * sizeof(float));
        float *scores = malloc((size_t)nh * sl * sizeof(float));
        float *weights = malloc((size_t)nh * sl * sizeof(float));
        float *output = malloc((size_t)nh * hd * sizeof(float));
        for (int i = 0; i < nh * hd; i++) queries[i] = 0.1f;
        for (int i = 0; i < nh * sl; i++) weights[i] = 1.0f / sl;

        bench_ctx_t bctx = {cache, queries, scores, weights, output,
                            nh, sl, hd};
        double t_k = bench("attention_scores (1 layer, all heads)",
                           bench_k_scores, &bctx, 5, 20);
        bench("attention_output (1 layer, all heads)",
              bench_v_output, &bctx, 5, 20);

        /* Baseline: dequant + naive dot product */
        size_t f32_k_size = (size_t)nh * sl * hd;
        float *f32_buf = malloc(f32_k_size * sizeof(float));
        baseline_ctx_t blctx = {cache, queries, scores, f32_buf,
                                nh, sl, hd};
        double t_bl = bench("baseline: dequant+dot (1L, all heads)",
                            bench_baseline_k_scores, &blctx, 5, 20);

        /* Pure f32 baseline */
        f32_ctx_t fctx = {data, queries, scores, nh, sl, hd};
        double t_f32 = bench("baseline: pure f32 dot (1L, all heads)",
                             bench_f32_k_scores, &fctx, 5, 20);

        free(f32_buf);

        printf("\n  --- K-score comparison ---\n");
        printf("  fused Q4 (eakv):       %8.0f us\n", t_k);
        printf("  dequant + dot:         %8.0f us (%.1fx slower)\n",
               t_bl, t_bl / t_k);
        printf("  pure f32 dot:          %8.0f us (%.1fx slower)\n",
               t_f32, t_f32 / t_k);

        double k_bytes = (double)nh * sl * 32 * 2;
        double k_gbps = k_bytes / (t_k * 1e-6) / 1e9;
        printf("  K-score throughput:                      "
               "%.1f GB/s (packed Q4)\n", k_gbps);

        const char *path = "/tmp/bench_libeakv.eakv";
        io_ctx_t ictx = { cache, path };
        bench("save .eakv", bench_save, &ictx, 1, 5);
        bench("load .eakv", bench_load, &ictx, 1, 5);

        long orig = (long)nl * 2 * nh * sl * hd * 4;
        float ratio = eakv_cache_compression_ratio(cache);
        printf("\n  f32 size:  %.1f MB\n", orig / (1024.0 * 1024.0));
        printf("  Q4 size:   %.1f MB (%.1fx compression)\n",
               orig * ratio / (1024.0 * 1024.0), 1.0 / ratio);

        free(data); free(queries); free(scores); free(weights); free(output);
        eakv_cache_free(cache);
        unlink(path);
    }

    printf("\n");
    return 0;
}
