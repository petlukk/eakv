# llama.cpp + eakv Integration Test Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Benchmark llama.cpp KV cache with F16 vs Q4_1, then validate eakv kernels against real model KV data.

**Architecture:** Build stock llama.cpp, run TinyLlama 1.1B with F16 and Q4_1 KV caches, measure memory/speed/output. Then write a C test that extracts KV data from a llama.cpp session, loads it into eakv, and compares attention results against ggml's Q4_1.

**Tech Stack:** llama.cpp (CMake, CPU-only), libeakv, TinyLlama-1.1B-Chat-v1.0.Q4_K_M.gguf

---

### Task 1: Build llama.cpp

**Step 1: Clone llama.cpp**

```bash
cd /root/dev
git clone --depth 1 https://github.com/ggerganov/llama.cpp
cd llama.cpp
```

**Step 2: Build CPU-only**

```bash
cmake -B build -DGGML_AVX512=ON
cmake --build build --config Release -j1
```

Note: `-j1` because we only have 1 core and 3.8GB RAM. Higher parallelism will OOM.

**Step 3: Verify build**

```bash
./build/bin/llama-cli --version
```

Expected: prints version info.

**Step 4: Commit note**

No commit needed — llama.cpp is a dependency, not our code.

---

### Task 2: Download TinyLlama model

**Step 1: Download the GGUF model**

```bash
cd /root/dev/llama.cpp
./build/bin/llama-cli -hf TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF -hff tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf -p "test" -n 1
```

This downloads and runs 1 token to verify it works. The GGUF file will be cached in `~/.cache/llama.cpp/` or downloaded to the current directory.

Alternatively, if `-hf` doesn't work:

```bash
curl -L -o tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf \
  "https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"
```

**Step 2: Verify model loads**

```bash
./build/bin/llama-cli -m tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf -p "Hello" -n 1
```

Expected: prints model info and generates 1 token.

---

### Task 3: Baseline benchmark — F16 KV cache

**Step 1: Run with F16 KV cache (default)**

```bash
/usr/bin/time -v ./build/bin/llama-cli \
  -m tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf \
  -p "Explain quantum computing in simple terms." \
  -n 256 \
  --cache-type-k f16 \
  --cache-type-v f16 \
  -c 2048 \
  --no-warmup \
  2>&1 | tee /root/dev/eakv/bench_f16.log
```

**Step 2: Record metrics from output**

From llama.cpp output, note:
- `llama_perf_context_print` lines: `eval time`, `sampling time`, tokens/second
- From `/usr/bin/time -v`: `Maximum resident set size (kbytes)`

Save these numbers.

---

### Task 4: Benchmark — Q4_1 KV cache

**Step 1: Run with Q4_1 KV cache**

```bash
/usr/bin/time -v ./build/bin/llama-cli \
  -m tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf \
  -p "Explain quantum computing in simple terms." \
  -n 256 \
  --cache-type-k q4_1 \
  --cache-type-v q4_1 \
  -c 2048 \
  --no-warmup \
  2>&1 | tee /root/dev/eakv/bench_q4_1.log
```

**Step 2: Record same metrics**

Note: Q4_1 requires flash attention in some llama.cpp versions. If it errors, add `--flash-attn` flag:

```bash
/usr/bin/time -v ./build/bin/llama-cli \
  -m tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf \
  -p "Explain quantum computing in simple terms." \
  -n 256 \
  --cache-type-k q4_1 \
  --cache-type-v q4_1 \
  --flash-attn \
  -c 2048 \
  --no-warmup \
  2>&1 | tee /root/dev/eakv/bench_q4_1.log
```

---

### Task 5: Compare outputs visually

**Step 1: Run both side by side with deterministic settings**

```bash
# F16 KV
./build/bin/llama-cli \
  -m tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf \
  -p "Explain quantum computing in simple terms." \
  -n 256 \
  --cache-type-k f16 --cache-type-v f16 \
  -c 2048 --seed 42 --temp 0 \
  2>/dev/null > /root/dev/eakv/output_f16.txt

# Q4_1 KV
./build/bin/llama-cli \
  -m tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf \
  -p "Explain quantum computing in simple terms." \
  -n 256 \
  --cache-type-k q4_1 --cache-type-v q4_1 \
  -c 2048 --seed 42 --temp 0 \
  2>/dev/null > /root/dev/eakv/output_q4_1.txt
```

**Step 2: Diff the outputs**

```bash
diff /root/dev/eakv/output_f16.txt /root/dev/eakv/output_q4_1.txt
```

Note differences. Some divergence is expected since Q4 is lossy.

---

### Task 6: Extract KV data and validate eakv kernels

This is the key task that proves eakv works on real model data.

**Step 1: Write a C program that extracts KV cache from llama.cpp**

Create: `/root/dev/eakv/tests/test_llama_kv.c`

This program:
1. Uses llama.cpp's C API to load a model, run a prompt, and extract the raw KV cache data
2. Loads the same data into eakv via `eakv_cache_load_raw()`
3. Runs `eakv_attention_scores()` with random queries
4. Compares against a naive f32 reference implementation
5. Reports max error and whether results are within Q4 tolerance

```c
#include "llama.h"
#include "eakv.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

// Extract raw f32 KV cache from llama.cpp context
// Layout: [n_layers][2][n_kv_heads][seq_len][head_dim]
static float *extract_kv_f32(struct llama_context *ctx,
                              const struct llama_model *model,
                              int n_layers, int n_kv_heads,
                              int head_dim, int seq_len) {
    int n_embd_gqa = n_kv_heads * head_dim;
    size_t total = (size_t)n_layers * 2 * n_kv_heads * seq_len * head_dim;
    float *data = malloc(total * sizeof(float));
    if (!data) return NULL;

    // llama.cpp stores K as [n_embd_gqa, kv_size] in F16
    // We need to read it out via the state API or direct tensor access
    // For now, use llama_state_seq_get_data to get per-sequence KV data
    // and manually reformat

    // TODO: This depends on llama.cpp's internal API for tensor access.
    // The exact extraction method needs to be determined at implementation
    // time by checking what's available in the llama.h public API.

    return data;
}

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <model.gguf>\n", argv[0]);
        return 1;
    }

    // Load model
    struct llama_model_params mparams = llama_model_default_params();
    struct llama_model *model = llama_model_load_from_file(argv[1], mparams);

    struct llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx = 512;  // small context for test
    struct llama_context *ctx = llama_init_from_model(model, cparams);

    // Run a short prompt to populate KV cache
    // ... tokenize, eval ...

    // Extract KV data
    // ... extract and compare with eakv ...

    printf("KV extraction and eakv comparison complete.\n");

    llama_free(ctx);
    llama_model_free(model);
    return 0;
}
```

**Note:** The exact implementation depends on what llama.cpp's public C API exposes for KV cache access. At implementation time, check `include/llama.h` for:
- `llama_state_seq_get_data` — serializes KV cache for a sequence
- Direct tensor access via `llama_get_kv_cache` or similar
- Or use the state save/load API to dump raw KV data

**Step 2: Build and link against both llama.cpp and libeakv**

```bash
gcc -O2 -I/root/dev/llama.cpp/include -I/root/dev/eakv/include \
  tests/test_llama_kv.c \
  /root/dev/llama.cpp/build/src/libllama.a \
  /root/dev/llama.cpp/build/ggml/src/libggml.a \
  /root/dev/eakv/build/libeakv.a \
  -lm -lstdc++ -lpthread \
  -o build/test_llama_kv
```

**Step 3: Run and verify**

```bash
./build/test_llama_kv tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf
```

Expected: prints max error between eakv attention and reference, shows it's within Q4 tolerance (~0.1 or less).

---

### Task 7: Write comparison report

**Step 1: Create results document**

Create: `/root/dev/eakv/docs/plans/2026-03-10-llama-benchmark-results.md`

Contents:

```markdown
# llama.cpp KV Cache Benchmark Results

## Setup
- CPU: AMD EPYC 9354P (1 core)
- RAM: 3.8 GB
- Model: TinyLlama 1.1B Q4_K_M
- Context: 2048
- Generated: 256 tokens
- Prompt: "Explain quantum computing in simple terms."

## Memory

| KV Cache Type | Peak RSS |
|---|---|
| F16 (default) | ??? MB |
| Q4_1 | ??? MB |
| Savings | ??? MB |

## Speed

| KV Cache Type | Prompt eval (tok/s) | Generation (tok/s) |
|---|---|---|
| F16 | ??? | ??? |
| Q4_1 | ??? | ??? |

## Output Quality

[Visual comparison notes]

## eakv Kernel Validation

[Max error vs reference on real KV data]

## Conclusions

[Summary of findings]
```

**Step 2: Commit results**

```bash
cd /root/dev/eakv
git add docs/plans/2026-03-10-llama-benchmark-results.md bench_f16.log bench_q4_1.log output_f16.txt output_q4_1.txt
git commit -m "bench: llama.cpp KV cache F16 vs Q4_1 comparison with TinyLlama 1.1B"
```
