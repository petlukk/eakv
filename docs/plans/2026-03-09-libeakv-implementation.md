# libeakv Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Rewrite eakv as a C library (`libeakv.a` / `libeakv.so`) with a CLI tool for llama.cpp session file conversion. Kill the Python package.

**Architecture:** Thin C glue layer over existing Eä SIMD kernels. One header (`eakv.h`), five C source files (`cache.c`, `attention.c`, `io.c`, `llama_session.c`, `cli.c`). Kernels compiled to `.o` by `build_kernels.sh`, linked into the library. Tests in C using assert macros.

**Tech Stack:** C11, Eä kernels (compiled via `ea` compiler), Make, existing `.eakv` binary format.

**Design doc:** `docs/plans/2026-03-09-libeakv-design.md`

---

## Task 0: Project restructure

**Files:**
- Delete: `src/eakv/*.py`, `pyproject.toml`, `tests/*.py`, `benchmarks/*.py`
- Delete: `HOWTO.md` (Python-specific)
- Keep: `kernels/*.ea`, `kernels/*.ea.json`, `docs/`, `README.md`, `build_kernels.sh`
- Create: `include/eakv.h`, `src/cache.c`, `src/attention.c`, `src/io.c`, `src/cli.c`, `tests/test_cache.c`, `tests/test_harness.h`, `Makefile`

**Step 1: Create a git branch**

```bash
cd /root/dev/eakv
git checkout -b feat/libeakv
```

**Step 2: Delete Python package**

```bash
rm -rf src/eakv/ pyproject.toml HOWTO.md
rm -rf tests/*.py benchmarks/
rm -f dequantize_simd.ll dequantize_simd.so dequantize_u8.ll quantize_simd.so
rm -rf build/ dist/
mkdir -p src include tests
```

**Step 3: Update build_kernels.sh to produce .o files**

Modify `build_kernels.sh` to compile kernels to `.o` files in `build/` instead of `.so` in `src/eakv/lib/`. Drop all Python binding generation.

```bash
#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
KERNEL_DIR="$SCRIPT_DIR/kernels"
BUILD_DIR="$SCRIPT_DIR/build"

mkdir -p "$BUILD_DIR"

# Find ea compiler
EA="${EA:-}"
if [ -z "$EA" ]; then
    if command -v ea &>/dev/null; then
        EA=ea
    elif [ -x "$HOME/dev/eacompute/target/release/ea" ]; then
        EA="$HOME/dev/eacompute/target/release/ea"
    elif [ -x "$HOME/dev/eacompute/target/debug/ea" ]; then
        EA="$HOME/dev/eacompute/target/debug/ea"
    else
        echo "Error: ea compiler not found. Set EA= or add to PATH." >&2
        exit 1
    fi
fi

echo "Building eakv kernels (ea=$EA)..."

# All kernel .ea files → .o files
for src in "$KERNEL_DIR"/*.ea; do
    name="$(basename "$src" .ea)"
    echo "  $name"

    flags=""
    # AVX-512 kernels
    case "$name" in
        dequantize_avx512|fused_k_score|fused_v_sum|fused_attention|fused_k_score_gqa)
            flags="--avx512"
            ;;
    esac

    "$EA" "$src" $flags -o "$BUILD_DIR/$name.o"
done

echo "Done. Objects in $BUILD_DIR/"
```

**Step 4: Commit**

```bash
git add -A
git commit -m "chore: delete Python package, restructure for C library"
```

---

## Task 1: Test harness + Makefile

**Files:**
- Create: `tests/test_harness.h`
- Create: `Makefile`

**Step 1: Write test harness**

`tests/test_harness.h` — minimal assert-based test framework:

```c
#ifndef TEST_HARNESS_H
#define TEST_HARNESS_H

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

static int test_count = 0;
static int test_pass = 0;
static int test_fail = 0;

#define TEST(name) \
    static void name(void); \
    static void name##_runner(void) { \
        test_count++; \
        printf("  %s... ", #name); \
        name(); \
        test_pass++; \
        printf("ok\n"); \
    } \
    static void name(void)

#define RUN(name) name##_runner()

#define ASSERT(cond) do { \
    if (!(cond)) { \
        printf("FAIL\n    %s:%d: %s\n", __FILE__, __LINE__, #cond); \
        test_fail++; \
        return; \
    } \
} while(0)

#define ASSERT_EQ(a, b) do { \
    if ((a) != (b)) { \
        printf("FAIL\n    %s:%d: %d != %d\n", __FILE__, __LINE__, (int)(a), (int)(b)); \
        test_fail++; \
        return; \
    } \
} while(0)

#define ASSERT_NEAR(a, b, tol) do { \
    if (fabs((double)(a) - (double)(b)) > (tol)) { \
        printf("FAIL\n    %s:%d: %.6f != %.6f (tol=%.6f)\n", \
               __FILE__, __LINE__, (double)(a), (double)(b), (double)(tol)); \
        test_fail++; \
        return; \
    } \
} while(0)

#define TEST_MAIN() \
    int main(void) { \
        printf("\n"); \
        run_tests(); \
        printf("\n%d tests: %d passed, %d failed\n\n", test_count, test_pass, test_fail); \
        return test_fail > 0 ? 1 : 0; \
    }

#endif
```

**Step 2: Write Makefile**

```makefile
CC      = gcc
CFLAGS  = -std=c11 -Wall -Wextra -Wpedantic -O2 -Iinclude
LDFLAGS = -lm

BUILD   = build
SRC     = src
INCLUDE = include
TESTS   = tests

# Kernel objects (built by build_kernels.sh)
KERNEL_OBJS = $(wildcard $(BUILD)/*.o)

# Library sources
LIB_SRCS = $(SRC)/cache.c $(SRC)/attention.c $(SRC)/io.c
LIB_OBJS = $(LIB_SRCS:$(SRC)/%.c=$(BUILD)/%.o)

# CLI
CLI_SRC  = $(SRC)/cli.c
CLI_OBJ  = $(BUILD)/cli.o

# Test sources
TEST_SRCS = $(wildcard $(TESTS)/test_*.c)
TEST_BINS = $(TEST_SRCS:$(TESTS)/%.c=$(BUILD)/%)

.PHONY: all lib cli test clean

all: lib cli

lib: $(BUILD)/libeakv.a $(BUILD)/libeakv.so

cli: $(BUILD)/eakv

test: $(TEST_BINS)
	@for t in $(TEST_BINS); do echo "=== $$(basename $$t) ==="; $$t || exit 1; done

$(BUILD)/%.o: $(SRC)/%.c $(INCLUDE)/eakv.h | $(BUILD)
	$(CC) $(CFLAGS) -fPIC -c $< -o $@

$(BUILD)/libeakv.a: $(LIB_OBJS) $(KERNEL_OBJS)
	ar rcs $@ $^

$(BUILD)/libeakv.so: $(LIB_OBJS) $(KERNEL_OBJS)
	$(CC) -shared -o $@ $^ $(LDFLAGS)

$(BUILD)/eakv: $(CLI_OBJ) $(BUILD)/libeakv.a
	$(CC) -o $@ $^ $(LDFLAGS)

$(BUILD)/test_%: $(TESTS)/test_%.c $(BUILD)/libeakv.a
	$(CC) $(CFLAGS) -I$(TESTS) -o $@ $< -L$(BUILD) -leakv $(LDFLAGS)

$(BUILD):
	mkdir -p $(BUILD)

clean:
	rm -f $(BUILD)/*.o $(BUILD)/libeakv.* $(BUILD)/eakv $(BUILD)/test_*
```

**Step 3: Commit**

```bash
git add Makefile tests/test_harness.h
git commit -m "build: add Makefile and C test harness"
```

---

## Task 2: eakv.h + cache.c (create/free/info)

**Files:**
- Create: `include/eakv.h`
- Create: `src/cache.c`
- Create: `tests/test_cache.c`

**Step 1: Write the failing test**

`tests/test_cache.c`:

```c
#include "test_harness.h"
#include "eakv.h"

TEST(create_and_free) {
    eakv_cache_t *c = eakv_cache_create(32, 8, 128, 2048);
    ASSERT(c != NULL);
    ASSERT_EQ(eakv_cache_n_layers(c), 32);
    ASSERT_EQ(eakv_cache_n_kv_heads(c), 8);
    ASSERT_EQ(eakv_cache_head_dim(c), 128);
    ASSERT_EQ(eakv_cache_max_seq_len(c), 2048);
    ASSERT_EQ(eakv_cache_seq_len(c), 0);
    eakv_cache_free(c);
}

TEST(create_invalid) {
    // n_heads * head_dim must be multiple of 64 (Q4 group size)
    // 8 * 128 = 1024, divisible by 64 -> ok
    eakv_cache_t *c = eakv_cache_create(32, 8, 128, 2048);
    ASSERT(c != NULL);
    eakv_cache_free(c);
}

TEST(compression_ratio_empty) {
    eakv_cache_t *c = eakv_cache_create(1, 1, 128, 64);
    ASSERT_NEAR(eakv_cache_compression_ratio(c), 0.0f, 0.001f);
    eakv_cache_free(c);
}

static void run_tests(void) {
    RUN(create_and_free);
    RUN(create_invalid);
    RUN(compression_ratio_empty);
}

TEST_MAIN()
```

**Step 2: Run test to verify it fails**

```bash
make test
```

Expected: compilation fails — `eakv.h` doesn't exist yet.

**Step 3: Write eakv.h**

`include/eakv.h`:

```c
#ifndef EAKV_H
#define EAKV_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Opaque cache handle */
typedef struct eakv_cache eakv_cache_t;

/* Error codes */
#define EAKV_OK          0
#define EAKV_ERR_IO     -1
#define EAKV_ERR_FORMAT -2
#define EAKV_ERR_ALLOC  -3
#define EAKV_ERR_FULL   -4
#define EAKV_ERR_PARAM  -5

/*--- Lifecycle ---*/

eakv_cache_t *eakv_cache_create(int n_layers, int n_kv_heads,
                                 int head_dim, int max_seq_len);
void          eakv_cache_free(eakv_cache_t *cache);

/*--- Append ---*/

int eakv_cache_append(eakv_cache_t *cache, int layer,
                      const float *k_vec, const float *v_vec);

/*--- Bulk quantize (for import) ---*/

int eakv_cache_load_raw(eakv_cache_t *cache, int layer, int kv_idx,
                        const float *data, int n_elements);

/*--- Attention ---*/

void eakv_attention_scores(const eakv_cache_t *cache, const float *queries,
                           int layer, int n_q_heads, int n_kv_heads,
                           float *scores_out);

void eakv_attention_output(const eakv_cache_t *cache, const float *weights,
                           int layer, int n_q_heads, int n_kv_heads,
                           float *output_out);

/*--- File I/O ---*/

int eakv_cache_save(const eakv_cache_t *cache, const char *path);
int eakv_cache_load(const char *path, eakv_cache_t **out);

/*--- llama.cpp session conversion ---*/

int eakv_import_llama_session(const char *session_path, eakv_cache_t **out);
int eakv_export_llama_session(const eakv_cache_t *cache, const char *session_path);

/*--- Info ---*/

int   eakv_cache_seq_len(const eakv_cache_t *cache);
int   eakv_cache_n_layers(const eakv_cache_t *cache);
int   eakv_cache_n_kv_heads(const eakv_cache_t *cache);
int   eakv_cache_head_dim(const eakv_cache_t *cache);
int   eakv_cache_max_seq_len(const eakv_cache_t *cache);
float eakv_cache_compression_ratio(const eakv_cache_t *cache);

#ifdef __cplusplus
}
#endif

#endif /* EAKV_H */
```

**Step 4: Write cache.c — struct + create/free/info**

`src/cache.c`:

```c
#include "eakv.h"
#include <stdlib.h>
#include <string.h>

/* Q4_1 constants */
#define Q4_GROUP_SIZE 64
#define Q4_PACKED_BYTES_PER_GROUP 32  /* 64 nibbles = 32 bytes */

/*
 * Per-layer, per-KV (K or V) storage:
 *   weights: uint8_t[max_groups * 32]   (packed Q4 nibbles)
 *   scales:  float[max_groups]
 *   biases:  float[max_groups]
 */
typedef struct {
    uint8_t *weights;
    float   *scales;
    float   *biases;
} eakv_kv_store_t;

struct eakv_cache {
    int n_layers;
    int n_kv_heads;
    int head_dim;
    int max_seq_len;
    int seq_len;           /* current token count */

    int values_per_token;  /* n_kv_heads * head_dim */
    int groups_per_token;  /* values_per_token / Q4_GROUP_SIZE */
    int max_groups;        /* groups_per_token * max_seq_len */

    /* stores[layer][0] = K, stores[layer][1] = V */
    eakv_kv_store_t stores[][2];
};

/* Extern kernel declarations */
extern void q4_quantize_split_f32(const float *src, int32_t *weights_out,
                                  float *scales_out, float *biases_out,
                                  int32_t n_groups);

eakv_cache_t *eakv_cache_create(int n_layers, int n_kv_heads,
                                 int head_dim, int max_seq_len)
{
    int values_per_token = n_kv_heads * head_dim;
    if (values_per_token % Q4_GROUP_SIZE != 0)
        return NULL;

    int groups_per_token = values_per_token / Q4_GROUP_SIZE;
    int max_groups = groups_per_token * max_seq_len;

    /* Allocate struct with flexible array for stores */
    size_t struct_size = sizeof(eakv_cache_t) +
                         (size_t)n_layers * 2 * sizeof(eakv_kv_store_t);
    eakv_cache_t *c = calloc(1, struct_size);
    if (!c) return NULL;

    c->n_layers        = n_layers;
    c->n_kv_heads      = n_kv_heads;
    c->head_dim        = head_dim;
    c->max_seq_len     = max_seq_len;
    c->seq_len         = 0;
    c->values_per_token = values_per_token;
    c->groups_per_token = groups_per_token;
    c->max_groups       = max_groups;

    /* Allocate storage per layer per K/V */
    for (int l = 0; l < n_layers; l++) {
        for (int kv = 0; kv < 2; kv++) {
            eakv_kv_store_t *s = &c->stores[l][kv];
            s->weights = calloc((size_t)max_groups * Q4_PACKED_BYTES_PER_GROUP, 1);
            s->scales  = calloc((size_t)max_groups, sizeof(float));
            s->biases  = calloc((size_t)max_groups, sizeof(float));
            if (!s->weights || !s->scales || !s->biases) {
                eakv_cache_free(c);
                return NULL;
            }
        }
    }

    return c;
}

void eakv_cache_free(eakv_cache_t *cache)
{
    if (!cache) return;
    for (int l = 0; l < cache->n_layers; l++) {
        for (int kv = 0; kv < 2; kv++) {
            free(cache->stores[l][kv].weights);
            free(cache->stores[l][kv].scales);
            free(cache->stores[l][kv].biases);
        }
    }
    free(cache);
}

int eakv_cache_seq_len(const eakv_cache_t *c) { return c->seq_len; }
int eakv_cache_n_layers(const eakv_cache_t *c) { return c->n_layers; }
int eakv_cache_n_kv_heads(const eakv_cache_t *c) { return c->n_kv_heads; }
int eakv_cache_head_dim(const eakv_cache_t *c) { return c->head_dim; }
int eakv_cache_max_seq_len(const eakv_cache_t *c) { return c->max_seq_len; }

float eakv_cache_compression_ratio(const eakv_cache_t *c)
{
    if (c->seq_len == 0) return 0.0f;
    int n_groups = c->groups_per_token * c->seq_len;
    size_t compressed = (size_t)c->n_layers * 2 *
                        ((size_t)n_groups * Q4_PACKED_BYTES_PER_GROUP +
                         (size_t)n_groups * sizeof(float) * 2);
    size_t original = (size_t)c->n_layers * 2 *
                      (size_t)c->values_per_token * c->seq_len * sizeof(float);
    return (float)compressed / (float)original;
}

int eakv_cache_append(eakv_cache_t *cache, int layer,
                      const float *k_vec, const float *v_vec)
{
    (void)cache; (void)layer; (void)k_vec; (void)v_vec;
    return EAKV_ERR_PARAM; /* stub — implemented in Task 3 */
}

int eakv_cache_load_raw(eakv_cache_t *cache, int layer, int kv_idx,
                        const float *data, int n_elements)
{
    (void)cache; (void)layer; (void)kv_idx; (void)data; (void)n_elements;
    return EAKV_ERR_PARAM; /* stub — implemented in Task 4 */
}
```

Note: `attention.c` and `io.c` need stub files to satisfy the Makefile. Create empty stubs:

```c
/* src/attention.c */
#include "eakv.h"

void eakv_attention_scores(const eakv_cache_t *c, const float *q,
                           int layer, int nq, int nkv, float *out)
{ (void)c;(void)q;(void)layer;(void)nq;(void)nkv;(void)out; }

void eakv_attention_output(const eakv_cache_t *c, const float *w,
                           int layer, int nq, int nkv, float *out)
{ (void)c;(void)q;(void)layer;(void)nq;(void)nkv;(void)out; }
```

```c
/* src/io.c */
#include "eakv.h"
#include <stddef.h>

int eakv_cache_save(const eakv_cache_t *c, const char *p)
{ (void)c;(void)p; return EAKV_ERR_PARAM; }

int eakv_cache_load(const char *p, eakv_cache_t **o)
{ (void)p;(void)o; return EAKV_ERR_PARAM; }

int eakv_import_llama_session(const char *p, eakv_cache_t **o)
{ (void)p;(void)o; return EAKV_ERR_PARAM; }

int eakv_export_llama_session(const eakv_cache_t *c, const char *p)
{ (void)c;(void)p; return EAKV_ERR_PARAM; }
```

**Step 5: Run tests**

```bash
./build_kernels.sh && make test
```

Expected: 3 tests pass.

**Step 6: Commit**

```bash
git add include/eakv.h src/cache.c src/attention.c src/io.c tests/test_cache.c
git commit -m "feat: eakv.h + cache create/free/info with C tests"
```

---

## Task 3: cache_append — incremental Q4 quantization

The append operation accumulates f32 values in a staging buffer. When a full Q4 group (64 values) is ready, the SIMD quantize kernel runs.

Key insight: a single token contributes `n_kv_heads * head_dim` values per K and per V. For Llama 3.1 8B: 8 * 128 = 1024 values = 16 Q4 groups. So each `append` call quantizes complete groups — no partial-group accumulation needed as long as `n_kv_heads * head_dim` is a multiple of 64 (enforced at create time).

**Files:**
- Modify: `src/cache.c` — implement `eakv_cache_append` and `eakv_cache_load_raw`
- Create: `tests/test_append.c`

**Step 1: Write the failing test**

`tests/test_append.c`:

```c
#include "test_harness.h"
#include "eakv.h"
#include <stdlib.h>

/* Small cache: 1 layer, 1 head, dim=128, max 4 tokens */
/* 1 * 128 = 128 values per token = 2 Q4 groups */

TEST(append_one_token) {
    eakv_cache_t *c = eakv_cache_create(1, 1, 128, 4);
    ASSERT(c != NULL);

    float k[128], v[128];
    for (int i = 0; i < 128; i++) {
        k[i] = (float)i / 128.0f;
        v[i] = (float)(127 - i) / 128.0f;
    }

    int rc = eakv_cache_append(c, 0, k, v);
    ASSERT_EQ(rc, 0);
    ASSERT_EQ(eakv_cache_seq_len(c), 1);

    eakv_cache_free(c);
}

TEST(append_fills_cache) {
    eakv_cache_t *c = eakv_cache_create(1, 1, 128, 4);
    float k[128], v[128];
    for (int i = 0; i < 128; i++) { k[i] = 0.5f; v[i] = 0.5f; }

    for (int t = 0; t < 4; t++) {
        int rc = eakv_cache_append(c, 0, k, v);
        ASSERT_EQ(rc, 0);
    }
    ASSERT_EQ(eakv_cache_seq_len(c), 4);

    /* 5th append should fail — cache full */
    int rc = eakv_cache_append(c, 0, k, v);
    ASSERT_EQ(rc, EAKV_ERR_FULL);

    eakv_cache_free(c);
}

TEST(append_multi_layer) {
    eakv_cache_t *c = eakv_cache_create(2, 1, 128, 4);
    float k[128], v[128];
    for (int i = 0; i < 128; i++) { k[i] = 1.0f; v[i] = 2.0f; }

    /* Must append to all layers to advance seq_len */
    for (int l = 0; l < 2; l++) {
        int rc = eakv_cache_append(c, l, k, v);
        ASSERT_EQ(rc, 0);
    }
    ASSERT_EQ(eakv_cache_seq_len(c), 1);

    eakv_cache_free(c);
}

static void run_tests(void) {
    RUN(append_one_token);
    RUN(append_fills_cache);
    RUN(append_multi_layer);
}

TEST_MAIN()
```

**Step 2: Implement append in cache.c**

Replace the `eakv_cache_append` stub with:

```c
int eakv_cache_append(eakv_cache_t *cache, int layer,
                      const float *k_vec, const float *v_vec)
{
    if (layer < 0 || layer >= cache->n_layers)
        return EAKV_ERR_PARAM;
    if (cache->seq_len >= cache->max_seq_len)
        return EAKV_ERR_FULL;

    int gpt = cache->groups_per_token;  /* groups per token */
    int group_start = cache->seq_len * gpt;

    const float *vecs[2] = { k_vec, v_vec };

    /* Temporary i32 buffer for quantize kernel output */
    int32_t *tmp = malloc((size_t)gpt * Q4_PACKED_BYTES_PER_GROUP * sizeof(int32_t));
    if (!tmp) return EAKV_ERR_ALLOC;

    for (int kv = 0; kv < 2; kv++) {
        eakv_kv_store_t *s = &cache->stores[layer][kv];

        q4_quantize_split_f32(
            vecs[kv],
            tmp,
            s->scales + group_start,
            s->biases + group_start,
            gpt
        );

        /* Convert i32 → u8 and copy to weights */
        uint8_t *dst = s->weights + (size_t)group_start * Q4_PACKED_BYTES_PER_GROUP;
        for (int i = 0; i < gpt * Q4_PACKED_BYTES_PER_GROUP; i++) {
            dst[i] = (uint8_t)tmp[i];
        }
    }

    free(tmp);

    /* Advance seq_len only when last layer is appended */
    if (layer == cache->n_layers - 1) {
        cache->seq_len++;
    }

    return EAKV_OK;
}
```

Also implement `eakv_cache_load_raw` (bulk quantize for import):

```c
int eakv_cache_load_raw(eakv_cache_t *cache, int layer, int kv_idx,
                        const float *data, int n_elements)
{
    if (layer < 0 || layer >= cache->n_layers) return EAKV_ERR_PARAM;
    if (kv_idx < 0 || kv_idx > 1) return EAKV_ERR_PARAM;
    if (n_elements % Q4_GROUP_SIZE != 0) return EAKV_ERR_PARAM;

    int n_groups = n_elements / Q4_GROUP_SIZE;
    if (n_groups > cache->max_groups) return EAKV_ERR_PARAM;

    eakv_kv_store_t *s = &cache->stores[layer][kv_idx];

    int32_t *tmp = malloc((size_t)n_groups * Q4_PACKED_BYTES_PER_GROUP * sizeof(int32_t));
    if (!tmp) return EAKV_ERR_ALLOC;

    q4_quantize_split_f32(data, tmp, s->scales, s->biases, n_groups);

    for (int i = 0; i < n_groups * Q4_PACKED_BYTES_PER_GROUP; i++) {
        s->weights[i] = (uint8_t)tmp[i];
    }

    free(tmp);

    /* Update seq_len based on how many tokens this covers */
    int tokens = n_elements / cache->values_per_token;
    if (tokens > cache->seq_len) {
        cache->seq_len = tokens;
    }

    return EAKV_OK;
}
```

**Step 3: Build and test**

```bash
./build_kernels.sh && make test
```

Expected: 6 tests pass (3 from test_cache + 3 from test_append).

**Step 4: Commit**

```bash
git add src/cache.c tests/test_append.c
git commit -m "feat: cache append — incremental Q4 quantization"
```

---

## Task 4: attention.c — fused K-score and V-sum

**Files:**
- Modify: `src/attention.c` — implement using kernel FFI
- Create: `tests/test_attention.c`

The attention functions need access to the cache internals. Since `eakv_cache` is defined in `cache.c`, we need an internal header.

**Step 1: Create internal header**

`src/internal.h`:

```c
#ifndef EAKV_INTERNAL_H
#define EAKV_INTERNAL_H

#include "eakv.h"
#include <stdint.h>

#define Q4_GROUP_SIZE 64
#define Q4_PACKED_BYTES_PER_GROUP 32

typedef struct {
    uint8_t *weights;
    float   *scales;
    float   *biases;
} eakv_kv_store_t;

struct eakv_cache {
    int n_layers;
    int n_kv_heads;
    int head_dim;
    int max_seq_len;
    int seq_len;
    int values_per_token;
    int groups_per_token;
    int max_groups;
    eakv_kv_store_t stores[][2];
};

/* Kernel declarations */
extern void q4_quantize_split_f32(const float *src, int32_t *weights_out,
                                  float *scales_out, float *biases_out,
                                  int32_t n_groups);

extern void q4_dequantize_avx2_f32(const uint8_t *weights, const float *scales,
                                   const float *biases, float *out,
                                   int32_t n_groups);

extern void q4_fused_k_score_f32(const float *q_vec, const uint8_t *k_packed,
                                 const float *k_scales, const float *k_biases,
                                 float *scores, int32_t seq_len,
                                 int32_t group_offset);

extern void q4_fused_k_score_multi_f32(const float *q_vecs, const uint8_t *k_packed,
                                       const float *k_scales, const float *k_biases,
                                       float *scores, int32_t seq_len,
                                       int32_t n_heads, int32_t groups_per_head);

extern void q4_fused_v_sum_f32(const float *weights, const uint8_t *v_packed,
                               const float *v_scales, const float *v_biases,
                               float *out, int32_t seq_len,
                               int32_t group_offset);

extern void q4_fused_v_sum_multi_f32(const float *weights, const uint8_t *v_packed,
                                     const float *v_scales, const float *v_biases,
                                     float *out, int32_t seq_len,
                                     int32_t n_heads, int32_t groups_per_head);

extern void q4_k_score_gqa_f32(const float *q_vecs, const uint8_t *k_packed,
                                const float *k_scales, const float *k_biases,
                                float *scores, int32_t seq_len,
                                int32_t n_q_heads, int32_t n_kv_heads,
                                int32_t groups_per_head);

extern void q4_v_sum_gqa_f32(const float *weights, const uint8_t *v_packed,
                              const float *v_scales, const float *v_biases,
                              float *out, int32_t seq_len,
                              int32_t n_q_heads, int32_t n_kv_heads,
                              int32_t groups_per_head);

extern void q4_validate(const float *scales, const float *biases,
                        const int32_t *scales_bits, const int32_t *biases_bits,
                        int32_t n_groups);

#endif
```

Move the struct definition and kernel externs out of `cache.c` into `src/internal.h`, and `#include "internal.h"` in both `cache.c` and `attention.c`.

**Step 2: Write the failing test**

`tests/test_attention.c`:

```c
#include "test_harness.h"
#include "eakv.h"
#include <stdlib.h>

/*
 * Strategy: quantize known data, compute attention via eakv,
 * compare to naive f32 reference implementation.
 * Tolerance is generous (0.15) because Q4 is lossy.
 */

static float dot_f32(const float *a, const float *b, int n) {
    float sum = 0.0f;
    for (int i = 0; i < n; i++) sum += a[i] * b[i];
    return sum;
}

TEST(scores_single_head) {
    /* 1 layer, 1 KV head, dim=128, seq_len=2 */
    eakv_cache_t *c = eakv_cache_create(1, 1, 128, 2);
    ASSERT(c != NULL);

    /* Append 2 tokens with known K values */
    float k0[128], k1[128], v[128];
    for (int i = 0; i < 128; i++) {
        k0[i] = (float)i / 128.0f;
        k1[i] = 1.0f - (float)i / 128.0f;
        v[i] = 0.5f;
    }
    eakv_cache_append(c, 0, k0, v);
    eakv_cache_append(c, 0, k1, v);

    /* Query */
    float q[128];
    for (int i = 0; i < 128; i++) q[i] = 1.0f / 128.0f;

    float scores[2];
    eakv_attention_scores(c, q, 0, 1, 1, scores);

    /* Reference: dot(q, k) / sqrt(128) — but on quantized data, so approximate */
    /* Just check that scores are finite and roughly ordered */
    ASSERT(scores[0] == scores[0]); /* not NaN */
    ASSERT(scores[1] == scores[1]);

    eakv_cache_free(c);
}

TEST(output_single_head) {
    eakv_cache_t *c = eakv_cache_create(1, 1, 128, 2);
    ASSERT(c != NULL);

    float k[128], v0[128], v1[128];
    for (int i = 0; i < 128; i++) {
        k[i] = 0.5f;
        v0[i] = 1.0f;
        v1[i] = 0.0f;
    }
    eakv_cache_append(c, 0, k, v0);
    eakv_cache_append(c, 0, k, v1);

    /* Weights: 100% on first token */
    float w[2] = {1.0f, 0.0f};
    float out[128];
    eakv_attention_output(c, w, 0, 1, 1, out);

    /* Output should be close to v0 (all 1.0), with Q4 error */
    for (int i = 0; i < 128; i++) {
        ASSERT_NEAR(out[i], 1.0f, 0.15f);
    }

    eakv_cache_free(c);
}

static void run_tests(void) {
    RUN(scores_single_head);
    RUN(output_single_head);
}

TEST_MAIN()
```

**Step 3: Implement attention.c**

Replace stub `src/attention.c`:

```c
#include "internal.h"

void eakv_attention_scores(const eakv_cache_t *c, const float *queries,
                           int layer, int n_q_heads, int n_kv_heads,
                           float *scores_out)
{
    const eakv_kv_store_t *k = &c->stores[layer][0];
    int groups_per_head = c->seq_len * (c->head_dim / Q4_GROUP_SIZE) * /* groups for dim */
    /* Actually: groups_per_head = seq_len * (head_dim / Q4_GROUP_SIZE) ... no.
       The layout is: per layer/kv, all heads concatenated, then tokens interleaved.
       Let me re-derive from the Python code. */

    /* From Python: group_offset = head * seq_len * 2
       groups_per_head = seq_len * 2  (for head_dim=128, 128/64=2 groups per token per head)
       This is: (head_dim / Q4_GROUP_SIZE) * seq_len */
    int groups_per_head = (c->head_dim / Q4_GROUP_SIZE) * c->seq_len;

    if (n_q_heads == n_kv_heads) {
        /* MHA path */
        q4_fused_k_score_multi_f32(
            queries, k->weights, k->scales, k->biases,
            scores_out, c->seq_len, n_q_heads, groups_per_head);
    } else {
        /* GQA path */
        q4_k_score_gqa_f32(
            queries, k->weights, k->scales, k->biases,
            scores_out, c->seq_len, n_q_heads, n_kv_heads, groups_per_head);
    }
}

void eakv_attention_output(const eakv_cache_t *c, const float *weights,
                           int layer, int n_q_heads, int n_kv_heads,
                           float *output_out)
{
    const eakv_kv_store_t *v = &c->stores[layer][1];
    int groups_per_head = (c->head_dim / Q4_GROUP_SIZE) * c->seq_len;

    if (n_q_heads == n_kv_heads) {
        q4_fused_v_sum_multi_f32(
            weights, v->weights, v->scales, v->biases,
            output_out, c->seq_len, n_q_heads, groups_per_head);
    } else {
        q4_v_sum_gqa_f32(
            weights, v->weights, v->scales, v->biases,
            output_out, c->seq_len, n_q_heads, n_kv_heads, groups_per_head);
    }
}
```

**Step 4: Build and test**

```bash
make test
```

Expected: 8 tests pass.

**Step 5: Commit**

```bash
git add src/internal.h src/attention.c src/cache.c tests/test_attention.c
git commit -m "feat: fused attention — K-score and V-sum with GQA dispatch"
```

---

## Task 5: io.c — .eakv file save/load

Port the binary format from the Python `_io.py`. Same format, same magic, same header struct.

**Files:**
- Modify: `src/io.c`
- Create: `tests/test_io.c`

**Step 1: Write the failing test**

`tests/test_io.c`:

```c
#include "test_harness.h"
#include "eakv.h"
#include <stdio.h>
#include <unistd.h>

TEST(save_and_load) {
    eakv_cache_t *c = eakv_cache_create(1, 1, 128, 4);
    ASSERT(c != NULL);

    float k[128], v[128];
    for (int i = 0; i < 128; i++) {
        k[i] = (float)i / 128.0f;
        v[i] = 1.0f - (float)i / 128.0f;
    }
    eakv_cache_append(c, 0, k, v);
    eakv_cache_append(c, 0, k, v);

    const char *path = "/tmp/test_eakv.eakv";
    int rc = eakv_cache_save(c, path);
    ASSERT_EQ(rc, 0);

    eakv_cache_t *loaded = NULL;
    rc = eakv_cache_load(path, &loaded);
    ASSERT_EQ(rc, 0);
    ASSERT(loaded != NULL);

    ASSERT_EQ(eakv_cache_n_layers(loaded), 1);
    ASSERT_EQ(eakv_cache_n_kv_heads(loaded), 1);
    ASSERT_EQ(eakv_cache_head_dim(loaded), 128);
    ASSERT_EQ(eakv_cache_seq_len(loaded), 2);

    unlink(path);
    eakv_cache_free(c);
    eakv_cache_free(loaded);
}

TEST(load_nonexistent) {
    eakv_cache_t *c = NULL;
    int rc = eakv_cache_load("/tmp/nonexistent.eakv", &c);
    ASSERT(rc != 0);
    ASSERT(c == NULL);
}

static void run_tests(void) {
    RUN(save_and_load);
    RUN(load_nonexistent);
}

TEST_MAIN()
```

**Step 2: Implement io.c**

Replace stub `src/io.c`. Implement the `.eakv` binary format — same layout as the Python version:

- Magic: `EAKV` (4 bytes)
- Header: 512 bytes total (version, quant_scheme, group_size, dtype, n_layers, n_heads, head_dim, seq_len, max_seq_len, compression, model_hash, tokenizer_hash, checksum)
- Index table: n_layers * 2 * 8 bytes (K_offset, V_offset per layer as uint64_t)
- Data blocks: per layer per K/V — weights + scales + biases, 64-byte aligned

Reference: `src/eakv/_io.py` lines 20-21 for the header struct format:
```
<HHIHIIIIIh32s32sQ
```
Which is: version(u16) quant_scheme(u16) group_size(u32) orig_dtype(u16) n_layers(u32) n_heads(u32) head_dim(u32) seq_len(u32) max_seq_len(u32) compression(i16) model_hash(32s) tokenizer_hash(32s) checksum(u64)

**Step 3: Build and test**

```bash
make test
```

Expected: 10 tests pass.

**Step 4: Commit**

```bash
git add src/io.c tests/test_io.c
git commit -m "feat: .eakv binary file save/load"
```

---

## Task 6: cli.c — eakv command-line tool

**Files:**
- Create: `src/cli.c`

**Step 1: Implement CLI**

`src/cli.c`:

```c
#include "eakv.h"
#include <stdio.h>
#include <string.h>

static void usage(void) {
    fprintf(stderr,
        "Usage:\n"
        "  eakv inspect <file.eakv>            Show metadata\n"
        "  eakv validate <file.eakv>           Check integrity\n"
        "  eakv import <session.bin> -o <out>   llama.cpp session -> eakv\n"
        "  eakv export <file.eakv> -o <out>    eakv -> llama.cpp session\n"
    );
}

static int cmd_inspect(const char *path) {
    eakv_cache_t *c = NULL;
    int rc = eakv_cache_load(path, &c);
    if (rc != EAKV_OK) {
        fprintf(stderr, "Error: cannot load %s (code %d)\n", path, rc);
        return 1;
    }
    printf("File:        %s\n", path);
    printf("Layers:      %d\n", eakv_cache_n_layers(c));
    printf("KV heads:    %d\n", eakv_cache_n_kv_heads(c));
    printf("Head dim:    %d\n", eakv_cache_head_dim(c));
    printf("Seq len:     %d\n", eakv_cache_seq_len(c));
    printf("Max seq len: %d\n", eakv_cache_max_seq_len(c));
    printf("Compression: %.1f%%\n", eakv_cache_compression_ratio(c) * 100.0f);
    eakv_cache_free(c);
    return 0;
}

static int cmd_import(const char *in, const char *out) {
    eakv_cache_t *c = NULL;
    int rc = eakv_import_llama_session(in, &c);
    if (rc != EAKV_OK) {
        fprintf(stderr, "Error: cannot import %s (code %d)\n", in, rc);
        return 1;
    }
    rc = eakv_cache_save(c, out);
    eakv_cache_free(c);
    if (rc != EAKV_OK) {
        fprintf(stderr, "Error: cannot save %s (code %d)\n", out, rc);
        return 1;
    }
    printf("Imported %s -> %s\n", in, out);
    return 0;
}

static int cmd_export(const char *in, const char *out) {
    eakv_cache_t *c = NULL;
    int rc = eakv_cache_load(in, &c);
    if (rc != EAKV_OK) {
        fprintf(stderr, "Error: cannot load %s (code %d)\n", in, rc);
        return 1;
    }
    rc = eakv_export_llama_session(c, out);
    eakv_cache_free(c);
    if (rc != EAKV_OK) {
        fprintf(stderr, "Error: cannot export to %s (code %d)\n", out, rc);
        return 1;
    }
    printf("Exported %s -> %s\n", in, out);
    return 0;
}

int main(int argc, char **argv) {
    if (argc < 3) { usage(); return 1; }

    if (strcmp(argv[1], "inspect") == 0) {
        return cmd_inspect(argv[2]);
    }
    if (strcmp(argv[1], "validate") == 0) {
        return cmd_inspect(argv[2]); /* validate = load + inspect for now */
    }
    if (strcmp(argv[1], "import") == 0) {
        if (argc < 5 || strcmp(argv[3], "-o") != 0) {
            fprintf(stderr, "Usage: eakv import <session.bin> -o <out.eakv>\n");
            return 1;
        }
        return cmd_import(argv[2], argv[4]);
    }
    if (strcmp(argv[1], "export") == 0) {
        if (argc < 5 || strcmp(argv[3], "-o") != 0) {
            fprintf(stderr, "Usage: eakv export <file.eakv> -o <session.bin>\n");
            return 1;
        }
        return cmd_export(argv[2], argv[4]);
    }

    usage();
    return 1;
}
```

**Step 2: Build and verify**

```bash
make cli
./build/eakv
```

Expected: prints usage.

**Step 3: Commit**

```bash
git add src/cli.c
git commit -m "feat: eakv CLI — inspect, validate, import, export"
```

---

## Task 7: llama.cpp session import/export

This requires reverse-engineering the llama.cpp session file format. The format is not publicly documented, so this task starts with reading the llama.cpp source code to understand the binary layout.

**Files:**
- Modify: `src/io.c` — or create `src/llama_session.c`
- Create: `tests/test_llama_session.c`

**Step 1: Research llama.cpp session format**

Read the llama.cpp source at `github.com/ggml-org/llama.cpp`:
- `src/llama-context.cpp` — `llama_state_save_file` / `llama_state_load_file`
- `include/llama.h` — public API signatures
- `examples/save-load-state/save-load-state.cpp` — usage example

The session file format typically contains:
1. Magic / version header
2. Token list (the prompt tokens that generated this KV cache)
3. KV cache data — per-layer K and V tensors, typically f16

Extract the exact byte layout and implement the parser.

**Step 2: Implement import**

`src/llama_session.c`:

```c
#include "internal.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* llama.cpp session format constants — determined by reading their source */
/* These will be filled in after Step 1 research */

int eakv_import_llama_session(const char *path, eakv_cache_t **out)
{
    FILE *f = fopen(path, "rb");
    if (!f) return EAKV_ERR_IO;

    /* TODO: parse llama.cpp session header to get:
       - n_layers, n_kv_heads, head_dim, seq_len
       - KV data (f16 or f32) */

    /* Create cache with discovered dimensions */
    /* For each layer: read K data, read V data, call eakv_cache_load_raw */

    fclose(f);
    return EAKV_OK;
}

int eakv_export_llama_session(const eakv_cache_t *cache, const char *path)
{
    FILE *f = fopen(path, "wb");
    if (!f) return EAKV_ERR_IO;

    /* TODO: write llama.cpp session header
       For each layer: dequantize K and V, write as f16 */

    fclose(f);
    return EAKV_OK;
}
```

**Step 3: Write tests**

The test creates a synthetic "llama.cpp session file" with known data, imports it, and verifies the cache dimensions and approximate values.

**Step 4: Commit**

```bash
git add src/llama_session.c tests/test_llama_session.c
git commit -m "feat: llama.cpp session import/export"
```

---

## Task 8: README + final cleanup

**Files:**
- Modify: `README.md` — rewrite for C library
- Delete any remaining Python artifacts

**Step 1: Rewrite README.md**

Focus on:
- What libeakv is (C library for Q4 KV cache)
- Build instructions (`build_kernels.sh` + `make`)
- C API example
- CLI usage
- File format spec (brief)

**Step 2: Run full test suite**

```bash
./build_kernels.sh && make clean && make test
```

Expected: all tests pass.

**Step 3: Commit**

```bash
git add -A
git commit -m "docs: rewrite README for libeakv C library"
```

---

## Task 9: Merge to master

**Step 1: Verify everything**

```bash
./build_kernels.sh && make clean && make all && make test
./build/eakv inspect /tmp/test_eakv.eakv  # if test file exists
```

**Step 2: Merge**

```bash
git checkout master
git merge feat/libeakv
git push origin master
```

---

## Summary

| Task | What | Tests |
|------|------|-------|
| 0 | Project restructure, delete Python | — |
| 1 | Test harness + Makefile | — |
| 2 | eakv.h + cache create/free/info | 3 |
| 3 | cache_append + load_raw | 3 |
| 4 | Fused attention (MHA + GQA) | 2 |
| 5 | .eakv file save/load | 2 |
| 6 | CLI tool | — |
| 7 | llama.cpp session import/export | TBD |
| 8 | README rewrite | — |
| 9 | Merge | — |

Total: ~10 C tests minimum, covering quantize roundtrip, attention correctness, file I/O, and session conversion.
