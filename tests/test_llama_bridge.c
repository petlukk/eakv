/* Test eakv_from_llama_state() bridge function.
 *
 * Loads TinyLlama, runs a prompt, extracts state via llama.cpp,
 * feeds it through the bridge, and verifies attention accuracy.
 */
#define _POSIX_C_SOURCE 199309L
#include "llama.h"
#include "ggml.h"
#include "eakv.h"
#include "eakv_llama.h"
#include "internal.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

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

    printf("=== test_llama_bridge ===\n\n");

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

    printf("Model: %d layers, %d/%d heads, head_dim=%d\n", n_layer, n_head, n_head_kv, head_dim);

    const char *prompt = "The quick brown fox jumps over the lazy dog. "
                         "Quantum computing uses qubits that can exist in "
                         "superposition, enabling parallel computation.";
    int max_tokens = 256;
    llama_token *tokens = malloc(max_tokens * sizeof(llama_token));
    const struct llama_vocab *vocab = llama_model_get_vocab(model);
    int n_tokens = llama_tokenize(vocab, prompt, strlen(prompt),
                                   tokens, max_tokens, true, true);
    printf("Prompt: %d tokens\n", n_tokens);

    struct llama_batch batch = llama_batch_get_one(tokens, n_tokens);
    if (llama_decode(ctx, batch) != 0) { fprintf(stderr, "Decode failed\n"); return 1; }

    /* Extract state */
    size_t state_size = llama_state_seq_get_size(ctx, 0);
    uint8_t *state_buf = malloc(state_size);
    size_t written = llama_state_seq_get_data(ctx, state_buf, state_size, 0);
    if (written == 0) { fprintf(stderr, "State extraction failed\n"); return 1; }
    printf("State: %zu bytes\n\n", written);

    /* Use the bridge */
    eakv_cache_t *cache = NULL;
    int rc = eakv_from_llama_state(state_buf, written, n_layer, n_head_kv, head_dim, 512, &cache);
    if (rc != EAKV_OK) {
        fprintf(stderr, "Bridge failed: %d\n", rc);
        return 1;
    }
    printf("Bridge: created cache, seq_len=%d\n", eakv_cache_seq_len(cache));

    /* Verify: run attention on layer 0 with native head_dim */
    int sl = eakv_cache_seq_len(cache);
    float *queries = malloc((size_t)n_head_kv * head_dim * sizeof(float));
    float *scores = malloc((size_t)n_head_kv * sl * sizeof(float));
    for (int h = 0; h < n_head_kv; h++)
        for (int d = 0; d < head_dim; d++)
            queries[h * head_dim + d] = (float)((h * head_dim + d) * 7 % 1000) / 500.0f - 1.0f;

    eakv_attention_scores(cache, queries, 0, n_head_kv, n_head_kv, scores);

    /* Check for NaN */
    int n_nan = 0, n_finite = 0;
    for (int i = 0; i < n_head_kv * sl; i++) {
        if (!isfinite(scores[i])) n_nan++;
        else n_finite++;
    }
    printf("Scores: %d finite, %d NaN\n", n_finite, n_nan);

    /* Dequantize and compute reference */
    eakv_kv_data_t *k = &cache->kv[0];
    int gph = eakv_cache_max_seq_len(cache) * (head_dim / 64);
    float *k_f32 = malloc((size_t)n_head_kv * sl * head_dim * sizeof(float));
    for (int h = 0; h < n_head_kv; h++) {
        int n_groups = sl * (head_dim / 64);
        q4_dequantize_avx512_f32(
            k->weights + h * gph * 32,
            k->scales + h * gph,
            k->biases + h * gph,
            k_f32 + h * sl * head_dim,
            n_groups);
    }

    float *ref = malloc((size_t)n_head_kv * sl * sizeof(float));
    naive_k_scores_f32(queries, k_f32, ref, n_head_kv, sl, head_dim);

    double max_err = 0, sum_err = 0, sum_rel = 0;
    int n_rel = 0;
    for (int i = 0; i < n_head_kv * sl; i++) {
        double e = fabs(scores[i] - ref[i]);
        if (e > max_err) max_err = e;
        sum_err += e;
        if (fabs(ref[i]) > 0.01) { sum_rel += e / fabs(ref[i]); n_rel++; }
    }

    printf("Max abs error: %.4f\n", max_err);
    printf("Mean abs error: %.6f\n", sum_err / (n_head_kv * sl));
    printf("Mean rel error: %.2f%%\n\n", n_rel > 0 ? 100.0 * sum_rel / n_rel : 0.0);

    /* GQA test */
    float *gqa_scores = malloc((size_t)n_head * sl * sizeof(float));
    float *gqa_q = malloc((size_t)n_head * head_dim * sizeof(float));
    for (int i = 0; i < n_head * head_dim; i++)
        gqa_q[i] = (float)((i * 13) % 1000) / 500.0f - 1.0f;

    eakv_attention_scores(cache, gqa_q, 0, n_head, n_head_kv, gqa_scores);
    int gqa_nan = 0;
    for (int i = 0; i < n_head * sl; i++)
        if (!isfinite(gqa_scores[i])) gqa_nan++;

    printf("GQA (%d Q, %d KV): %s (%d NaN)\n",
           n_head, n_head_kv, gqa_nan == 0 ? "PASSED" : "FAILED", gqa_nan);

    int ok = (n_nan == 0 && gqa_nan == 0 && max_err < 1.0);
    printf("\n%s\n", ok ? "ALL PASSED" : "FAILED");

    free(queries); free(scores); free(k_f32); free(ref);
    free(gqa_q); free(gqa_scores);
    free(state_buf); free(tokens);
    eakv_cache_free(cache);
    llama_free(ctx);
    llama_model_free(model);

    return ok ? 0 : 1;
}
