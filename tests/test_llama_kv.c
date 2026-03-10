/* Test eakv with real KV cache data from llama.cpp.
 *
 * Extracts F16 KV cache from TinyLlama, converts to f32, and:
 * 1. Validates Q4 quantization quality on real data (quantize → dequantize)
 * 2. Benchmarks fused Q4 attention vs naive f32 dot product
 * 3. Tests with head_dim=128 (padded) for comparison
 * 4. Tests with native head_dim=64 (TinyLlama's actual dimensions)
 */
#define _POSIX_C_SOURCE 199309L
#include "llama.h"
#include "ggml.h"
#include "eakv.h"
#include "internal.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

static double now_us(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e6 + ts.tv_nsec / 1e3;
}

static void fp16_to_f32(const uint16_t *src, float *dst, int n) {
    for (int i = 0; i < n; i++)
        dst[i] = ggml_fp16_to_fp32(src[i]);
}

/* Scaled dot-product: scores = Q·K / √d (matching fused kernel behavior) */
static void naive_k_scores_f32(const float *queries, const float *k_data,
                                float *scores, int n_heads, int seq_len,
                                int head_dim) {
    float scale = 1.0f / sqrtf((float)head_dim);
    for (int h = 0; h < n_heads; h++) {
        const float *q = queries + h * head_dim;
        for (int t = 0; t < seq_len; t++) {
            const float *k = k_data + (h * seq_len + t) * head_dim;
            float sum = 0.0f;
            for (int d = 0; d < head_dim; d++)
                sum += q[d] * k[d];
            scores[h * seq_len + t] = sum * scale;
        }
    }
}

int main(int argc, char **argv) {
    const char *model_path = "/root/dev/llama.cpp/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf";
    if (argc > 1) model_path = argv[1];

    printf("=== eakv Real KV Data Test ===\n\n");

    /* Load model */
    ggml_backend_load_all();

    struct llama_model_params mparams = llama_model_default_params();
    struct llama_model *model = llama_model_load_from_file(model_path, mparams);
    if (!model) { fprintf(stderr, "Failed to load model\n"); return 1; }

    struct llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx = 512;
    cparams.n_batch = 512;
    cparams.type_k = GGML_TYPE_F16;
    cparams.type_v = GGML_TYPE_F16;
    cparams.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;

    struct llama_context *ctx = llama_init_from_model(model, cparams);
    if (!ctx) { fprintf(stderr, "Failed to create context\n"); return 1; }

    int n_layer = llama_model_n_layer(model);
    int n_head = llama_model_n_head(model);
    int n_head_kv = llama_model_n_head_kv(model);
    int n_embd = llama_model_n_embd(model);
    int head_dim = n_embd / n_head;
    int n_embd_k_gqa = n_head_kv * head_dim;

    printf("Model: %d layers, %d heads (%d KV), %d dim, head_dim=%d\n",
           n_layer, n_head, n_head_kv, n_embd, head_dim);
    printf("n_embd_k_gqa = %d\n\n", n_embd_k_gqa);

    /* Tokenize and run prompt */
    const char *prompt = "The quick brown fox jumps over the lazy dog. "
                         "Quantum computing uses qubits that can exist in "
                         "superposition, enabling parallel computation.";
    int max_tokens = 256;
    llama_token *tokens = malloc(max_tokens * sizeof(llama_token));
    const struct llama_vocab *vocab = llama_model_get_vocab(model);
    int n_tokens = llama_tokenize(vocab, prompt, strlen(prompt),
                                   tokens, max_tokens, true, true);
    if (n_tokens < 0) { fprintf(stderr, "Tokenization failed\n"); return 1; }
    printf("Prompt: %d tokens\n", n_tokens);

    struct llama_batch batch = llama_batch_get_one(tokens, n_tokens);
    if (llama_decode(ctx, batch) != 0) {
        fprintf(stderr, "Decode failed\n"); return 1;
    }
    printf("Decode complete. KV cache populated.\n\n");

    /* Extract state */
    size_t state_size = llama_state_seq_get_size(ctx, 0);
    uint8_t *state_buf = malloc(state_size);
    size_t written = llama_state_seq_get_data(ctx, state_buf, state_size, 0);
    if (written == 0) { fprintf(stderr, "State extraction failed\n"); return 1; }

    /* Parse state buffer */
    uint8_t *ptr = state_buf;
    uint32_t n_stream = *(uint32_t *)ptr; ptr += 4;
    uint32_t cell_count = *(uint32_t *)ptr; ptr += 4;
    printf("Extracted: %u cells from %u stream(s)\n", cell_count, n_stream);

    /* Skip cell metadata */
    for (uint32_t i = 0; i < cell_count; i++) {
        ptr += sizeof(int32_t);
        uint32_t n_seq_id = *(uint32_t *)ptr; ptr += 4;
        ptr += n_seq_id * sizeof(int32_t);
    }

    uint32_t v_trans = *(uint32_t *)ptr; ptr += 4;
    uint32_t file_n_layer = *(uint32_t *)ptr; ptr += 4;
    printf("v_trans=%u, n_layer=%u\n\n", v_trans, file_n_layer);

    int seq_len = (int)cell_count;
    size_t f32_layer_size = (size_t)n_embd_k_gqa * seq_len;

    /* Extract K data (all layers) */
    float *all_k_f32 = malloc(n_layer * f32_layer_size * sizeof(float));
    for (int l = 0; l < n_layer; l++) {
        int32_t k_type = *(int32_t *)ptr; ptr += 4;
        uint64_t k_size_row = *(uint64_t *)ptr; ptr += 8;
        (void)k_type;

        uint16_t *k_fp16 = (uint16_t *)ptr;
        float *k_dst = all_k_f32 + l * f32_layer_size;
        for (uint32_t cell = 0; cell < cell_count; cell++)
            fp16_to_f32(k_fp16 + cell * n_embd_k_gqa,
                       k_dst + cell * n_embd_k_gqa, n_embd_k_gqa);
        ptr += k_size_row * cell_count;
    }

    /* Skip V for now — we only need K for attention score tests */
    /* (V extraction works but we don't need it for this test) */

    /* ============================================================
     * TEST 1: Quantization quality on real KV data
     * ============================================================ */
    printf("=== TEST 1: Quantization Quality on Real KV Data ===\n\n");

    /* Transpose layer 0 K from [pos][head][dim] → [head][pos][dim] */
    float *k_transposed = malloc(f32_layer_size * sizeof(float));
    for (int h = 0; h < n_head_kv; h++)
        for (int t = 0; t < seq_len; t++)
            memcpy(k_transposed + (h * seq_len + t) * head_dim,
                   all_k_f32 + (t * n_head_kv + h) * head_dim,
                   head_dim * sizeof(float));

    /* Quantize via eakv */
    int n_groups = n_head_kv * seq_len * head_dim / 64;
    int32_t *q_weights = malloc(n_groups * 32 * sizeof(int32_t));
    float *q_scales = malloc(n_groups * sizeof(float));
    float *q_biases = malloc(n_groups * sizeof(float));
    q4_quantize_split_f32(k_transposed, q_weights, q_scales, q_biases, n_groups);

    /* Pack to uint8 then dequantize */
    uint8_t *packed = malloc(n_groups * 32);
    for (int i = 0; i < n_groups * 32; i++)
        packed[i] = (uint8_t)q_weights[i];

    float *k_dequantized = malloc(f32_layer_size * sizeof(float));
    q4_dequantize_avx512_f32(packed, q_scales, q_biases, k_dequantized, n_groups);

    /* Compare original vs round-tripped */
    double max_err = 0, sum_err = 0, sum_sq_err = 0;
    double sum_orig_sq = 0;
    for (int i = 0; i < (int)f32_layer_size; i++) {
        double err = fabs(k_transposed[i] - k_dequantized[i]);
        if (err > max_err) max_err = err;
        sum_err += err;
        sum_sq_err += err * err;
        sum_orig_sq += (double)k_transposed[i] * k_transposed[i];
    }
    double rmse = sqrt(sum_sq_err / f32_layer_size);
    double rms_orig = sqrt(sum_orig_sq / f32_layer_size);
    double snr_db = 20.0 * log10(rms_orig / rmse);

    printf("  Layer 0 K: %d values (%d groups of 64)\n", (int)f32_layer_size, n_groups);
    printf("  Original range: [%.3f, %.3f]\n",
           k_transposed[0], k_transposed[0]);  /* will compute properly below */

    float kmin = k_transposed[0], kmax = k_transposed[0];
    for (int i = 1; i < (int)f32_layer_size; i++) {
        if (k_transposed[i] < kmin) kmin = k_transposed[i];
        if (k_transposed[i] > kmax) kmax = k_transposed[i];
    }
    printf("  Original range: [%.3f, %.3f], RMS=%.4f\n", kmin, kmax, rms_orig);
    printf("  Max absolute error: %.4f\n", max_err);
    printf("  Mean absolute error: %.6f\n", sum_err / f32_layer_size);
    printf("  RMSE: %.6f\n", rmse);
    printf("  SNR: %.1f dB\n", snr_db);
    printf("  Relative RMSE: %.2f%%\n\n", 100.0 * rmse / rms_orig);

    /* Per-head stats */
    for (int h = 0; h < n_head_kv; h++) {
        double h_max = 0, h_sum = 0;
        int h_start = h * seq_len * head_dim;
        int h_count = seq_len * head_dim;
        for (int i = h_start; i < h_start + h_count; i++) {
            double e = fabs(k_transposed[i] - k_dequantized[i]);
            if (e > h_max) h_max = e;
            h_sum += e;
        }
        printf("  Head %d: max_err=%.4f, mean_err=%.6f\n",
               h, h_max, h_sum / h_count);
    }
    printf("\n");

    /* Show sample values */
    printf("  Sample values (head 0, pos 0, first 8 dims):\n");
    printf("  %8s %10s %10s %10s\n", "dim", "original", "Q4_deq", "error");
    for (int d = 0; d < 8; d++) {
        printf("  %8d %10.4f %10.4f %10.4f\n",
               d, k_transposed[d], k_dequantized[d],
               k_transposed[d] - k_dequantized[d]);
    }
    printf("\n");

    /* ============================================================
     * TEST 2: Attention with real-magnitude data, head_dim=128
     * ============================================================ */
    printf("=== TEST 2: Attention with Real Data (head_dim=128, aligned seq_len) ===\n\n");

    /* Use 32 of 33 tokens (power-of-2 aligned) and pad head_dim to 128 */
    int aligned_sl = 32;  /* drop last token for SIMD alignment */
    int padded_hd = 128;
    int save_seq_len = seq_len;
    seq_len = aligned_sl;
    size_t padded_per_lkv = (size_t)n_head_kv * seq_len * padded_hd;
    float *padded_data = calloc(1 * 2 * padded_per_lkv, sizeof(float));

    /* K: copy real 64 dims twice (dims 0-63 and 64-127) */
    for (int h = 0; h < n_head_kv; h++)
        for (int t = 0; t < seq_len; t++) {
            float *dst = padded_data + (h * seq_len + t) * padded_hd;
            float *src = k_transposed + (h * seq_len + t) * head_dim;
            memcpy(dst, src, head_dim * sizeof(float));
            memcpy(dst + head_dim, src, head_dim * sizeof(float));
        }
    /* V: duplicate same data for completeness */
    memcpy(padded_data + padded_per_lkv, padded_data,
           padded_per_lkv * sizeof(float));

    eakv_cache_t *padded_cache = eakv_cache_create(1, n_head_kv, padded_hd, seq_len);
    if (!padded_cache) { fprintf(stderr, "cache create failed\n"); return 1; }
    int rc = eakv_cache_load_raw(padded_cache, padded_data, seq_len);
    if (rc != 0) { fprintf(stderr, "load_raw failed: %d\n", rc); return 1; }

    /* Queries: duplicate 64 real dims to match padded K */
    float *pq = calloc(n_head_kv * padded_hd, sizeof(float));
    for (int h = 0; h < n_head_kv; h++)
        for (int d = 0; d < head_dim; d++) {
            float v = (float)((h * head_dim + d) * 7 % 1000) / 500.0f - 1.0f;
            pq[h * padded_hd + d] = v;
            pq[h * padded_hd + head_dim + d] = v;
        }

    float *scores_q4 = calloc(n_head_kv * seq_len, sizeof(float));
    float *scores_ref = calloc(n_head_kv * seq_len, sizeof(float));

    eakv_attention_scores(padded_cache, pq, 0, n_head_kv, n_head_kv, scores_q4);
    naive_k_scores_f32(pq, padded_data, scores_ref, n_head_kv, seq_len, padded_hd);

    /* Compare */
    int n_nan = 0, n_finite = 0;
    double attn_max_err = 0, attn_sum_err = 0;
    double attn_sum_rel = 0;
    int n_rel = 0;
    for (int i = 0; i < n_head_kv * seq_len; i++) {
        if (!isfinite(scores_q4[i])) { n_nan++; continue; }
        n_finite++;
        double e = fabs(scores_q4[i] - scores_ref[i]);
        if (e > attn_max_err) attn_max_err = e;
        attn_sum_err += e;
        if (fabs(scores_ref[i]) > 0.01) {
            attn_sum_rel += e / fabs(scores_ref[i]);
            n_rel++;
        }
    }

    printf("  Config: %d heads, %d dim (padded), %d seq_len\n",
           n_head_kv, padded_hd, seq_len);
    printf("  Scores: %d total, %d finite, %d NaN\n",
           n_head_kv * seq_len, n_finite, n_nan);
    printf("  Max absolute error: %.4f\n", attn_max_err);
    printf("  Mean absolute error: %.6f\n",
           n_finite > 0 ? attn_sum_err / n_finite : 0.0);
    printf("  Mean relative error: %.2f%%\n\n",
           n_rel > 0 ? 100.0 * attn_sum_rel / n_rel : 0.0);

    for (int h = 0; h < n_head_kv; h++) {
        printf("  Head %d (first 5 positions):\n", h);
        for (int t = 0; t < 5 && t < seq_len; t++) {
            int idx = h * seq_len + t;
            printf("    pos %d: Q4=%.4f  f32=%.4f  err=%.4f\n",
                   t, scores_q4[idx], scores_ref[idx],
                   scores_q4[idx] - scores_ref[idx]);
        }
    }
    printf("\n");

    /* ============================================================
     * TEST 3: Speed benchmark with real-magnitude data
     * ============================================================ */
    printf("=== TEST 3: Speed Benchmark (real-magnitude, head_dim=128) ===\n\n");

    /* Warmup */
    for (int i = 0; i < 20; i++) {
        eakv_attention_scores(padded_cache, pq, 0, n_head_kv, n_head_kv, scores_q4);
        naive_k_scores_f32(pq, padded_data, scores_ref, n_head_kv, seq_len, padded_hd);
    }

    int runs = 200;
    double *t_q4_arr = malloc(runs * sizeof(double));
    double *t_f32_arr = malloc(runs * sizeof(double));

    for (int i = 0; i < runs; i++) {
        double t0 = now_us();
        eakv_attention_scores(padded_cache, pq, 0, n_head_kv, n_head_kv, scores_q4);
        t_q4_arr[i] = now_us() - t0;

        t0 = now_us();
        naive_k_scores_f32(pq, padded_data, scores_ref, n_head_kv, seq_len, padded_hd);
        t_f32_arr[i] = now_us() - t0;
    }

    for (int i = 0; i < runs - 1; i++)
        for (int j = i + 1; j < runs; j++) {
            if (t_q4_arr[j] < t_q4_arr[i]) { double t = t_q4_arr[i]; t_q4_arr[i] = t_q4_arr[j]; t_q4_arr[j] = t; }
            if (t_f32_arr[j] < t_f32_arr[i]) { double t = t_f32_arr[i]; t_f32_arr[i] = t_f32_arr[j]; t_f32_arr[j] = t; }
        }

    double med_q4 = t_q4_arr[runs / 2];
    double med_f32 = t_f32_arr[runs / 2];

    printf("  Fused Q4 (eakv):       %6.0f us\n", med_q4);
    printf("  Pure f32 dot product:  %6.0f us\n", med_f32);
    printf("  Speedup:               %.1fx\n\n", med_f32 / med_q4);

    /* Memory */
    seq_len = save_seq_len;  /* restore for memory calc */
    long f32_bytes = (long)n_layer * 2 * n_embd_k_gqa * seq_len * 4;
    printf("  Memory (full model, %d layers):\n", n_layer);
    printf("    F16 KV: %.2f MB  (llama.cpp default)\n", f32_bytes / 2.0 / (1024.0 * 1024.0));
    printf("    F32 KV: %.2f MB\n", f32_bytes / (1024.0 * 1024.0));
    printf("    Q4 KV:  %.2f MB  (eakv)\n",
           f32_bytes * eakv_cache_compression_ratio(padded_cache) / (1024.0 * 1024.0));
    printf("    Ratio:  %.1fx vs F16\n\n",
           2.0 / eakv_cache_compression_ratio(padded_cache));

    /* ============================================================
     * TEST 4: Native head_dim=64 attention (no padding!)
     * ============================================================ */
    printf("=== TEST 4: Native head_dim=64 Attention ===\n\n");

    int native_sl = 32;
    size_t native_per_lkv = (size_t)n_head_kv * native_sl * head_dim;
    float *native_data = calloc(1 * 2 * native_per_lkv, sizeof(float));

    /* K: use real data directly (already in [head][pos][dim] layout) */
    for (int h = 0; h < n_head_kv; h++)
        for (int t = 0; t < native_sl; t++)
            memcpy(native_data + (h * native_sl + t) * head_dim,
                   k_transposed + (h * save_seq_len + t) * head_dim,
                   head_dim * sizeof(float));
    /* V: same data */
    memcpy(native_data + native_per_lkv, native_data,
           native_per_lkv * sizeof(float));

    eakv_cache_t *native_cache = eakv_cache_create(1, n_head_kv, head_dim, native_sl);
    if (!native_cache) { fprintf(stderr, "native cache create failed\n"); return 1; }
    rc = eakv_cache_load_raw(native_cache, native_data, native_sl);
    if (rc != 0) { fprintf(stderr, "native load_raw failed: %d\n", rc); return 1; }

    /* Queries: random-ish values */
    float *nq = calloc(n_head_kv * head_dim, sizeof(float));
    for (int h = 0; h < n_head_kv; h++)
        for (int d = 0; d < head_dim; d++)
            nq[h * head_dim + d] = (float)((h * head_dim + d) * 7 % 1000) / 500.0f - 1.0f;

    float *native_q4 = calloc(n_head_kv * native_sl, sizeof(float));
    float *native_ref = calloc(n_head_kv * native_sl, sizeof(float));

    eakv_attention_scores(native_cache, nq, 0, n_head_kv, n_head_kv, native_q4);
    naive_k_scores_f32(nq, native_data, native_ref, n_head_kv, native_sl, head_dim);

    int n4_nan = 0, n4_finite = 0;
    double n4_max_err = 0, n4_sum_err = 0, n4_sum_rel = 0;
    int n4_rel = 0;
    for (int i = 0; i < n_head_kv * native_sl; i++) {
        if (!isfinite(native_q4[i])) { n4_nan++; continue; }
        n4_finite++;
        double e = fabs(native_q4[i] - native_ref[i]);
        if (e > n4_max_err) n4_max_err = e;
        n4_sum_err += e;
        if (fabs(native_ref[i]) > 0.01) {
            n4_sum_rel += e / fabs(native_ref[i]);
            n4_rel++;
        }
    }

    printf("  Config: %d heads, %d dim (NATIVE), %d seq_len\n",
           n_head_kv, head_dim, native_sl);
    printf("  Scores: %d total, %d finite, %d NaN\n",
           n_head_kv * native_sl, n4_finite, n4_nan);
    printf("  Max absolute error: %.4f\n", n4_max_err);
    printf("  Mean absolute error: %.6f\n",
           n4_finite > 0 ? n4_sum_err / n4_finite : 0.0);
    printf("  Mean relative error: %.2f%%\n\n",
           n4_rel > 0 ? 100.0 * n4_sum_rel / n4_rel : 0.0);

    for (int h = 0; h < n_head_kv; h++) {
        printf("  Head %d (first 5 positions):\n", h);
        for (int t = 0; t < 5 && t < native_sl; t++) {
            int idx = h * native_sl + t;
            printf("    pos %d: Q4=%.4f  f32=%.4f  err=%.4f\n",
                   t, native_q4[idx], native_ref[idx],
                   native_q4[idx] - native_ref[idx]);
        }
    }
    printf("\n");

    /* GQA test with native head_dim=64 */
    printf("  GQA test: %d Q heads, %d KV heads, head_dim=%d\n", n_head, n_head_kv, head_dim);
    float *gqa_q = calloc(n_head * head_dim, sizeof(float));
    float *gqa_scores = calloc(n_head * native_sl, sizeof(float));
    for (int h = 0; h < n_head; h++)
        for (int d = 0; d < head_dim; d++)
            gqa_q[h * head_dim + d] = (float)((h * head_dim + d) * 13 % 1000) / 500.0f - 1.0f;

    eakv_attention_scores(native_cache, gqa_q, 0, n_head, n_head_kv, gqa_scores);

    int gqa_nan = 0, gqa_ok = 0;
    for (int i = 0; i < n_head * native_sl; i++) {
        if (!isfinite(gqa_scores[i])) gqa_nan++;
        else gqa_ok++;
    }
    printf("  GQA scores: %d finite, %d NaN\n", gqa_ok, gqa_nan);
    if (gqa_nan == 0)
        printf("  GQA head_dim=64: PASSED\n\n");
    else
        printf("  GQA head_dim=64: FAILED (%d NaN)\n\n", gqa_nan);

    /* Speed: native vs padded */
    printf("  --- Native head_dim=64 speed ---\n");
    for (int i = 0; i < 50; i++)
        eakv_attention_scores(native_cache, nq, 0, n_head_kv, n_head_kv, native_q4);
    int n_runs = 200;
    double *t_native = malloc(n_runs * sizeof(double));
    for (int i = 0; i < n_runs; i++) {
        double t0 = now_us();
        eakv_attention_scores(native_cache, nq, 0, n_head_kv, n_head_kv, native_q4);
        t_native[i] = now_us() - t0;
    }
    for (int i = 0; i < n_runs - 1; i++)
        for (int j = i + 1; j < n_runs; j++)
            if (t_native[j] < t_native[i]) { double t = t_native[i]; t_native[i] = t_native[j]; t_native[j] = t; }
    printf("  Fused Q4 (head_dim=64):  %.0f us\n", t_native[n_runs/2]);
    printf("  Fused Q4 (head_dim=128): %.0f us (padded, from TEST 3)\n", med_q4);
    printf("\n");

    free(t_native);
    free(gqa_q); free(gqa_scores);
    free(nq); free(native_q4); free(native_ref); free(native_data);
    eakv_cache_free(native_cache);

    /* Cleanup */
    free(t_q4_arr); free(t_f32_arr);
    free(pq); free(scores_q4); free(scores_ref);
    free(padded_data);
    free(k_transposed); free(k_dequantized);
    free(q_weights); free(q_scales); free(q_biases); free(packed);
    free(all_k_f32);
    free(state_buf); free(tokens);
    eakv_cache_free(padded_cache);
    llama_free(ctx);
    llama_model_free(model);

    printf("=== Done ===\n");
    return 0;
}
