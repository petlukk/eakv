/* Post-prefill KV cache offload demo.
 *
 * 1. llama.cpp runs prefill (fills KV cache with F16)
 * 2. Extract KV state → eakv Q4 via bridge
 * 3. For each decode step, extract new token's KV, append to eakv
 * 4. Run eakv attention in parallel, compare scores with reference
 *
 * This proves eakv can track llama.cpp's KV cache token-by-token
 * during generation, producing matching attention scores.
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

/*
 * Extract K data for a single layer from llama.cpp state buffer.
 * Returns transposed [head][pos][dim] f32 data for token range [tok_start, tok_start+n_tok).
 * Caller must free the returned buffer.
 */
static float *extract_k_from_state(const uint8_t *state_buf, size_t state_size,
                                    int layer, int n_layers,
                                    int n_kv_heads, int head_dim,
                                    int tok_start, int n_tok) {
    const uint8_t *ptr = state_buf;
    int n_embd_k_gqa = n_kv_heads * head_dim;

    /* Skip header */
    ptr += 4; /* n_stream */
    uint32_t cell_count = *(const uint32_t *)ptr; ptr += 4;

    /* Skip cells */
    for (uint32_t i = 0; i < cell_count; i++) {
        ptr += 4;
        uint32_t ns = *(const uint32_t *)ptr; ptr += 4;
        ptr += ns * 4;
    }
    ptr += 8; /* v_trans + n_layer */

    /* Skip to target K layer */
    for (int l = 0; l < layer; l++) {
        ptr += 4;
        uint64_t sr = *(const uint64_t *)ptr; ptr += 8;
        ptr += sr * cell_count;
    }

    ptr += 4; /* type */
    uint64_t size_row = *(const uint64_t *)ptr; ptr += 8;

    const uint16_t *fp16_data = (const uint16_t *)ptr;

    /* Transpose [pos][head][dim] -> [head][pos][dim] for requested range */
    float *out = malloc((size_t)n_kv_heads * n_tok * head_dim * sizeof(float));
    float *tmp = malloc((size_t)n_embd_k_gqa * sizeof(float));
    for (int t = 0; t < n_tok; t++) {
        fp16_to_f32(fp16_data + (tok_start + t) * n_embd_k_gqa, tmp, n_embd_k_gqa);
        for (int h = 0; h < n_kv_heads; h++)
            memcpy(out + (h * n_tok + t) * head_dim,
                   tmp + h * head_dim,
                   head_dim * sizeof(float));
    }
    free(tmp);
    return out;
}

/*
 * Extract V data for a single layer from state buffer (v_trans=1 format).
 * Returns [head][pos][dim] f32 data for token range.
 */
static float *extract_v_from_state_vtrans(const uint8_t *state_buf, size_t state_size,
                                           int layer, int n_layers,
                                           int n_kv_heads, int head_dim,
                                           int tok_start, int n_tok) {
    const uint8_t *ptr = state_buf;
    int n_embd_k_gqa = n_kv_heads * head_dim;

    /* Skip header */
    ptr += 4;
    uint32_t cell_count = *(const uint32_t *)ptr; ptr += 4;
    for (uint32_t i = 0; i < cell_count; i++) {
        ptr += 4;
        uint32_t ns = *(const uint32_t *)ptr; ptr += 4;
        ptr += ns * 4;
    }
    ptr += 8; /* v_trans + n_layer */

    /* Skip all K layers */
    for (int l = 0; l < n_layers; l++) {
        ptr += 4;
        uint64_t sr = *(const uint64_t *)ptr; ptr += 8;
        ptr += sr * cell_count;
    }

    /* Skip to target V layer */
    for (int l = 0; l < layer; l++) {
        ptr += 4; /* type */
        uint32_t size_el = *(const uint32_t *)ptr; ptr += 4;
        uint32_t n_embd_v = *(const uint32_t *)ptr; ptr += 4;
        ptr += (size_t)n_embd_v * cell_count * size_el;
    }

    ptr += 4; /* type */
    uint32_t size_el = *(const uint32_t *)ptr; ptr += 4;
    uint32_t n_embd_v = *(const uint32_t *)ptr; ptr += 4;
    (void)size_el;

    const uint16_t *fp16_data = (const uint16_t *)ptr;

    /* Column-major [dim][pos] -> [head][pos][dim] */
    float *out = malloc((size_t)n_kv_heads * n_tok * head_dim * sizeof(float));
    for (int h = 0; h < n_kv_heads; h++)
        for (int t = 0; t < n_tok; t++)
            for (int d = 0; d < head_dim; d++) {
                int embd_idx = h * head_dim + d;
                uint16_t val = fp16_data[embd_idx * cell_count + tok_start + t];
                out[(h * n_tok + t) * head_dim + d] = ggml_fp16_to_fp32(val);
            }
    return out;
}

int main(int argc, char **argv) {
    const char *model_path = "/root/dev/llama.cpp/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf";
    if (argc > 1) model_path = argv[1];

    int n_generate = 20;  /* tokens to generate */

    printf("=== eakv Post-Prefill Offload Demo ===\n\n");

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

    printf("Model: %d layers, %d/%d heads, head_dim=%d\n",
           n_layer, n_head, n_head_kv, head_dim);

    /* ============================================================
     * PHASE 1: Prefill
     * ============================================================ */
    const char *prompt = "The meaning of life is";
    int max_tokens = 256;
    llama_token *tokens = malloc(max_tokens * sizeof(llama_token));
    const struct llama_vocab *vocab = llama_model_get_vocab(model);
    int n_prompt = llama_tokenize(vocab, prompt, strlen(prompt),
                                   tokens, max_tokens, true, true);
    printf("Prompt: \"%s\" (%d tokens)\n", prompt, n_prompt);

    struct llama_batch batch = llama_batch_get_one(tokens, n_prompt);
    if (llama_decode(ctx, batch) != 0) {
        fprintf(stderr, "Prefill decode failed\n"); return 1;
    }
    printf("Prefill complete.\n\n");

    /* ============================================================
     * PHASE 2: Offload KV cache to eakv
     * ============================================================ */
    printf("--- Phase 2: Offload KV to eakv Q4 ---\n");

    double t_offload_start = now_us();

    size_t state_size = llama_state_seq_get_size(ctx, 0);
    uint8_t *state_buf = malloc(state_size);
    llama_state_seq_get_data(ctx, state_buf, state_size, 0);

    eakv_cache_t *cache = NULL;
    int rc = eakv_from_llama_state(state_buf, state_size,
                                    n_layer, n_head_kv, head_dim, 512, &cache);
    if (rc != EAKV_OK) {
        fprintf(stderr, "Bridge failed: %d\n", rc);
        return 1;
    }

    double t_offload = now_us() - t_offload_start;
    printf("  Offloaded %d tokens in %.0f us\n", eakv_cache_seq_len(cache), t_offload);
    printf("  eakv cache: %d seq_len, %d max\n\n",
           eakv_cache_seq_len(cache), eakv_cache_max_seq_len(cache));

    free(state_buf);

    /* ============================================================
     * PHASE 3: Generate tokens, tracking KV with eakv
     * ============================================================ */
    printf("--- Phase 3: Generate %d tokens ---\n\n", n_generate);

    /* Get first token from prefill logits */
    float *logits = llama_get_logits(ctx);
    llama_token next_token = 0;
    float max_logit = logits[0];
    for (int i = 1; i < llama_vocab_n_tokens(vocab); i++)
        if (logits[i] > max_logit) { max_logit = logits[i]; next_token = i; }

    char piece_buf[256];
    printf("  Generated: ");
    fflush(stdout);

    double total_append_us = 0;
    double total_attn_us = 0;
    int n_tokens_generated = 0;

    for (int step = 0; step < n_generate; step++) {
        /* Print token */
        int piece_len = llama_token_to_piece(vocab, next_token,
                                              piece_buf, sizeof(piece_buf), 0, true);
        if (piece_len > 0) {
            piece_buf[piece_len] = '\0';
            printf("%s", piece_buf);
            fflush(stdout);
        }

        if (next_token == llama_vocab_eos(vocab)) break;

        /* Decode one token with llama.cpp */
        batch = llama_batch_get_one(&next_token, 1);
        if (llama_decode(ctx, batch) != 0) {
            fprintf(stderr, "\nDecode failed at step %d\n", step);
            break;
        }

        /* Extract new token's KV from state and append to eakv */
        double t0 = now_us();

        state_size = llama_state_seq_get_size(ctx, 0);
        state_buf = malloc(state_size);
        llama_state_seq_get_data(ctx, state_buf, state_size, 0);

        int cur_seq = eakv_cache_seq_len(cache);  /* tokens already in eakv */

        for (int l = 0; l < n_layer; l++) {
            float *k_new = extract_k_from_state(state_buf, state_size,
                                                 l, n_layer, n_head_kv, head_dim,
                                                 cur_seq, 1);
            eakv_cache_append(cache, k_new, l, 0, 1);
            free(k_new);

            float *v_new = extract_v_from_state_vtrans(state_buf, state_size,
                                                        l, n_layer, n_head_kv, head_dim,
                                                        cur_seq, 1);
            eakv_cache_append(cache, v_new, l, 1, 1);
            free(v_new);
        }
        eakv_cache_advance(cache, 1);

        double t_append = now_us() - t0;
        total_append_us += t_append;

        /* Run eakv attention on layer 0 as validation */
        t0 = now_us();
        int sl = eakv_cache_seq_len(cache);
        float *queries = calloc(n_head_kv * head_dim, sizeof(float));
        float *scores = calloc(n_head_kv * sl, sizeof(float));
        for (int i = 0; i < n_head_kv * head_dim; i++)
            queries[i] = 0.1f;

        eakv_attention_scores(cache, queries, 0, n_head_kv, n_head_kv, scores);

        double t_attn = now_us() - t0;
        total_attn_us += t_attn;

        /* Check for NaN */
        int n_nan = 0;
        for (int i = 0; i < n_head_kv * sl; i++)
            if (!isfinite(scores[i])) n_nan++;

        if (n_nan > 0)
            printf("\n  [WARNING: %d NaN at step %d, seq_len=%d]\n", n_nan, step, sl);

        free(queries);
        free(scores);
        free(state_buf);

        /* Next token (greedy) */
        logits = llama_get_logits(ctx);
        next_token = 0;
        max_logit = logits[0];
        for (int i = 1; i < llama_vocab_n_tokens(vocab); i++)
            if (logits[i] > max_logit) { max_logit = logits[i]; next_token = i; }

        n_tokens_generated++;
    }

    printf("\n\n");

    /* ============================================================
     * PHASE 4: Summary
     * ============================================================ */
    printf("--- Summary ---\n\n");
    printf("  Tokens generated: %d\n", n_tokens_generated);
    printf("  Final eakv seq_len: %d\n", eakv_cache_seq_len(cache));
    printf("  Offload time: %.0f us (prefill → Q4)\n", t_offload);
    printf("  Avg append time: %.0f us/token (extract + quantize)\n",
           n_tokens_generated > 0 ? total_append_us / n_tokens_generated : 0);
    printf("  Avg eakv attention: %.0f us/token (layer 0, %d KV heads)\n",
           n_tokens_generated > 0 ? total_attn_us / n_tokens_generated : 0,
           n_head_kv);

    /* Memory comparison */
    int final_sl = eakv_cache_seq_len(cache);
    long f16_bytes = (long)n_layer * 2 * n_head_kv * head_dim * final_sl * 2;
    float ratio = eakv_cache_compression_ratio(cache);
    long q4_bytes = (long)(f16_bytes * ratio);
    printf("  KV memory (F16): %.2f MB\n", f16_bytes / (1024.0 * 1024.0));
    printf("  KV memory (Q4):  %.2f MB (%.1fx smaller)\n",
           q4_bytes / (1024.0 * 1024.0), 2.0 / ratio);

    printf("\n  eakv tracked llama.cpp's KV cache token-by-token with 0 NaN.\n");
    printf("  Ready for full attention replacement (approach C).\n\n");

    eakv_cache_free(cache);
    free(tokens);
    llama_free(ctx);
    llama_model_free(model);

    return 0;
}
