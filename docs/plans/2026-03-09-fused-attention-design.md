# Fused Attention-Dequantization Design

**Goal:** Compute attention scores and weighted V sums directly from packed Q4 KV cache, eliminating the intermediate f32 restore step.

**Motivation:** The dequantize kernels are memory-bound at ~24 GB/s (single-thread DRAM ceiling). The only remaining path to 2x improvement is eliminating a full memory pass. Currently, attention requires: restore Q4 to f32 buffer, then compute. Fused kernels dequantize inline during the dot product, so packed bytes are loaded once and never stored as f32.

## Architecture

Two Ea SIMD kernels, each fused with inline Q4 dequantization:

1. **K score kernel** — `dot(query, dequant(K[t]))` per sequence position
2. **V weighted sum kernel** — `sum_t(weight[t] * dequant(V[t]))` per dimension

Softmax happens between the two calls in Python/NumPy. This keeps each kernel simple and independently testable.

## Kernel Signatures

```
export func q4_fused_k_score_f32(
    q_vec:    *restrict f32,    // query vector [128]
    k_packed: *restrict u8,     // packed K weights [n_groups * 32]
    k_scales: *restrict f32,    // K scales [n_groups]
    k_biases: *restrict f32,    // K biases [n_groups]
    scores:   *mut f32,         // output scores [seq_len]
    seq_len:  i32               // must be multiple of 64
)

export func q4_fused_v_sum_f32(
    weights:  *restrict f32,    // softmax scores [seq_len]
    v_packed: *restrict u8,     // packed V weights [n_groups * 32]
    v_scales: *restrict f32,    // V scales [n_groups]
    v_biases: *restrict f32,    // V biases [n_groups]
    out_vec:  *mut f32,         // output vector [128]
    seq_len:  i32
)
```

## Parameters

- **Attention type:** MHA first, then GQA (GQA is MHA with ratio=1 — just an outer loop change)
- **Query shape:** Single-token decode (Q is one vector per head, shape [128])
- **Head granularity:** One head per kernel call, caller loops over heads
- **Accumulation precision:** FP32
- **Head dimension:** Fixed 128 (covers LLaMA 2/3, Mistral, Gemma, Qwen)
- **ISA:** AVX-512 primary, SSE/AVX2 fallbacks later

## Layout Mapping

KV data is stored as flat packed bytes per head. For head_dim=128, each sequence position spans 2 consecutive groups (128 / 64 = 2):

- Position t uses groups `t*2` and `t*2+1`
- Group `t*2` covers dimensions 0-63
- Group `t*2+1` covers dimensions 64-127

## SIMD Strategy (AVX-512, f32x16)

**K score — per group (64 elements):**

```
// Load 32 packed bytes, extract nibbles
p0 = load(k_packed, g*32)         // u8x16
p1 = load(k_packed, g*32 + 16)    // u8x16
lo0 = p0 .& mask15                // values 0-15
hi0 = p0 ./ div16                 // values 32-47
lo1 = p1 .& mask15                // values 16-31
hi1 = p1 ./ div16                 // values 48-63

// Dequantize
vs = splat(scale);  vb = splat(bias)
d0 = fma(widen_u8_f32x16(lo0), vs, vb)
d1 = fma(widen_u8_f32x16(lo1), vs, vb)
d2 = fma(widen_u8_f32x16(hi0), vs, vb)
d3 = fma(widen_u8_f32x16(hi1), vs, vb)

// Dot with Q
acc += reduce_add(d0 .* q0) + reduce_add(d1 .* q1)
     + reduce_add(d2 .* q2) + reduce_add(d3 .* q3)
```

4 packed loads, 4 Q loads, 4 widen, 4 FMA, 4 multiply, 4 reduce_add. Dequantized values never touch memory.

**V sum — per group (64 elements):**

Same dequant, but instead of dot product, scaled accumulation:

```
vw = splat(weight)
o0 = load(out_vec, o_off)
store(out_vec, o_off, fma(d0, vw, o0))   // out += dequant * weight
```

4 extra output loads + 4 stores, but the output vector is only 128 floats (512 bytes) — stays in L1 the entire time.

## Python API

```python
def attention_scores(bundle, query, layer, kv=0, head=0, tokens=None):
    """Compute attention scores directly from packed Q4 KV cache.
    Returns: scores [seq_len] f32 (pre-softmax, scaled by 1/sqrt(d))"""

def attention_output(bundle, weights, layer, kv=1, head=0, tokens=None):
    """Compute weighted V sum directly from packed Q4 KV cache.
    Returns: out [128] f32"""
```

Caller applies softmax between the two calls. The `tokens` parameter selects a subsequence (maps to group range offsets).

## Testing

Compare fused output against existing two-step path (restore then compute). Must match within f32 epsilon. Test cases: minimum sequence (64), typical (2048), partial tokens, zero weights, single position.

## Expected Gain

Eliminates one full memory pass over KV cache. For large sequences where the operation is memory-bound, this should approach 2x over restore-then-compute.
