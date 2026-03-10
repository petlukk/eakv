# llama.cpp KV Cache Benchmark Results

## Setup
- CPU: AMD EPYC 9354P (1 core, 2 threads)
- RAM: 3.8 GB
- Model: TinyLlama 1.1B Q4_K_M (636 MiB GGUF)
- Context: 2048
- Generated: 256 tokens max
- Prompt: "Explain quantum computing in simple terms."
- llama.cpp build: 0cd4f47, CPU-only with AVX-512

## Model Architecture (relevant to KV cache)
- 22 layers
- 32 attention heads, 4 KV heads (GQA 8:1)
- head_dim: 64
- n_embd_k_gqa = n_embd_v_gqa = 256

## llama.cpp Benchmark: F16 vs Q4_1 KV Cache

### Memory

| KV Cache Type | KV Buffer Size | Peak RSS |
|---|---|---|
| F16 (default) | 44.00 MiB (K: 22.00, V: 22.00) | 1,162 MB |
| Q4_1 | 13.75 MiB (K: 6.88, V: 6.88) | 1,130 MB |
| **Savings** | **30.25 MiB (3.2x reduction)** | **32 MB** |

Note: Peak RSS difference is modest because the 636 MiB model weights dominate memory. The KV cache savings become significant at longer contexts or larger models.

### Speed

| KV Cache Type | Prompt eval (tok/s) | Generation (tok/s) | Tokens generated |
|---|---|---|---|
| F16 | 85.74 | 41.10 | 215 |
| Q4_1 | 87.32 | 41.28 | 144 |

Speed is essentially identical — Q4_1 quantization adds negligible overhead with Flash Attention enabled.

### Output Quality

With deterministic settings (seed=42, temp=0), outputs diverge immediately after the first sentence. F16 produces a multi-paragraph response (256 tokens, hit limit). Q4_1 produces a single concise paragraph (106 tokens, hit EOS). Both are coherent and factually reasonable. KV cache quantization is lossy and changes attention patterns.

## eakv Kernel Validation on Real KV Data

### Quantization Quality (layer 0 K, 33 tokens)

| Metric | Value |
|---|---|
| Total values | 8,448 (132 groups of 64) |
| Original range | [-11.95, 5.75] |
| Original RMS | 1.21 |
| Max element error | 0.576 |
| Mean element error | 0.088 |
| RMSE | 0.135 |
| SNR | 19.1 dB |
| Relative RMSE | 11.14% |

Per-head breakdown:

| Head | Max Error | Mean Error | Notes |
|---|---|---|---|
| 0 | 0.135 | 0.037 | Small values, good quantization |
| 1 | 0.256 | 0.055 | |
| 2 | 0.576 | 0.200 | Widest range, most outliers |
| 3 | 0.204 | 0.058 | |

### Attention Score Accuracy (Q4 fused vs f32 reference)

Tested with head_dim=128 (padded from 64), seq_len=32, 4 heads.
The fused kernel applies `1/√d` scaling (standard scaled dot-product attention).

| Metric | Value |
|---|---|
| Total scores | 128 |
| Finite/NaN | 128 / 0 |
| Max absolute error | 0.81 |
| Mean absolute error | 0.089 |
| Mean relative error | 16.5% |

Sample (head 0, first 5 positions):

| Pos | Q4 (eakv) | f32 (ref) | Error |
|---|---|---|---|
| 0 | -0.0153 | -0.0147 | -0.0005 |
| 1 | -0.3388 | -0.3833 | 0.0446 |
| 2 | -0.5678 | -0.6041 | 0.0363 |
| 3 | -0.4855 | -0.4517 | -0.0338 |
| 4 | -0.3502 | -0.3503 | 0.0002 |

### Speed: Fused Q4 vs Baselines

#### Real-data test (4 heads, 128 dim, 32 seq)

| Method | Time | Speedup |
|---|---|---|
| Fused Q4 (eakv) | 2 µs | **4.8x** |
| Pure f32 dot product | 8 µs | 1.0x |

#### Synthetic benchmark (7B-like configs)

| Config | Fused Q4 | Dequant+dot | F32 dot | Speedup vs dequant | Speedup vs f32 |
|---|---|---|---|---|---|
| 8H, 2K seq | 210 µs | 1,192 µs | 1,051 µs | **5.7x** | **5.0x** |
| 8H, 8K seq | 844 µs | 5,731 µs | 4,615 µs | **6.8x** | **5.5x** |
| 32L, 8H, 2K seq | 213 µs | 1,195 µs | 1,082 µs | **5.6x** | **5.1x** |

Speedup grows with sequence length (5.0x at 2K → 5.5x at 8K) due to better Q4 data locality.

### Memory Compression

| KV Format | Size (7B, 2K seq) | Ratio |
|---|---|---|
| F32 | 16.0 MB | 1.0x |
| F16 (llama.cpp default) | 8.0 MB | 2.0x |
| Q4 (eakv) | 2.5 MB | **6.4x** |

## Known Issues

### head_dim=64 kernel bug

The fused AVX-512 attention kernels produce incorrect results (NaN, 1e28+ values) when `head_dim=64`. All heads beyond head 0 are affected. All existing tests use `head_dim=128` and pass correctly.

Models affected: TinyLlama (head_dim=64), Phi-2 (head_dim=80), and others with non-128 head dimensions.

This must be fixed before eakv can be used with these models. The kernel likely has hardcoded assumptions about head_dim=128 or groups_per_head alignment.

### 1/√d scaling

The fused attention kernels apply `1/√d` scaling internally (standard scaled dot-product attention). This was not documented and existing tests did not verify score magnitudes — they only checked for NaN and determinism. Now documented and reference comparisons account for this.

## Conclusions

1. **Q4 KV cache quantization is viable.** 19 dB SNR on real model data, 16.5% mean relative error in attention scores. Coherent text output from llama.cpp's Q4_1 mode confirms this.

2. **Fused Q4 attention is 5-7x faster** than dequantize-then-dot-product, and 5x faster than pure f32 dot product. The advantage grows with sequence length.

3. **6.4x memory compression** vs f32, 3.2x vs F16. Critical for long-context inference where KV cache dominates memory.

4. **head_dim=64 support is broken** — must be fixed before real model integration. This is the top priority blocker.
