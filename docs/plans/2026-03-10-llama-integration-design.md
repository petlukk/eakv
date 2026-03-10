# llama.cpp + eakv Integration Test Design

**Goal:** Patch llama.cpp to use libeakv for KV cache storage. Compare memory, speed, and output quality against stock llama.cpp.

**Model:** TinyLlama 1.1B Q4_K_M GGUF (~640MB). 22 layers, 4 KV heads, 64 head_dim, 2048 default context.

**Approach:** Minimal surgical patch. No abstraction layer, no runtime switch. Replace F16 KV storage with eakv Q4 in the relevant structs and functions.

## What gets patched

llama.cpp stores KV cache as contiguous F16 tensors in `struct llama_kv_cache`. The patch:

1. Add libeakv as a dependency (static link `libeakv.a`)
2. Replace KV cache allocation with `eakv_cache_create()`
3. Replace KV cache writes (after each layer's attention) with quantization into eakv
4. Replace KV cache reads (during attention) with `eakv_attention_scores` / `eakv_attention_output`
5. Replace KV cache save/load with `.eakv` format

## Test protocol

Run both versions with identical settings:

```
Model: TinyLlama-1.1B-Chat-v1.0.Q4_K_M.gguf
Prompt: "Explain quantum computing in simple terms"
Context: 2048 tokens
Generate: 256 tokens
```

**Measure:**

- Peak RSS memory (via `/proc/self/status` or `/usr/bin/time -v`)
- Tokens/second (llama.cpp reports this)
- Output text (visual comparison)

**Expected results:**

- KV cache memory: ~4.5 MB (eakv Q4) vs ~28 MB (F16) for TinyLlama at 2K context
- Total RSS: meaningfully lower (model weights dominate, but KV savings grow with context)
- tok/s: similar or slightly slower (extra quantize step per token vs less memory pressure)
- Output: similar quality (Q4 KV is lossy but attention is approximate anyway)

## Deliverables

1. Patched llama.cpp fork in a local directory
2. Benchmark script that runs both versions and reports the comparison
3. Results captured in a doc

## Future (bigger machine)

- Larger model (7B/13B) where KV cache is a bigger fraction of total memory
- Longer contexts (8K+) where the 6.4x compression matters more
- Perplexity measurement for quantitative quality comparison
