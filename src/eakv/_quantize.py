"""High-level quantize API."""

import numpy as np
from numpy.typing import NDArray

from ._bundle import Q4Bundle
from ._ops import q4_quantize_f32


def quantize(kv_cache: NDArray) -> Q4Bundle:
    """Quantize a KV cache tensor to Q4_1.

    Args:
        kv_cache: shape (n_layers, 2, n_heads, seq_len, head_dim), dtype float32 or float16
                  Axis 1: 0=K, 1=V

    Returns:
        Q4Bundle containing the quantized data.
    """
    if kv_cache.ndim != 5:
        raise ValueError(
            f"Expected 5D tensor (n_layers, 2, n_heads, seq_len, head_dim), "
            f"got shape {kv_cache.shape}"
        )

    n_layers, kv, n_heads, seq_len, head_dim = kv_cache.shape
    if kv != 2:
        raise ValueError(f"Expected axis 1 to be 2 (K, V), got {kv}")

    orig_dtype = str(kv_cache.dtype)

    # Convert to f32 for quantization
    if kv_cache.dtype == np.float16:
        kv_f32 = kv_cache.astype(np.float32)
    elif kv_cache.dtype == np.float32:
        kv_f32 = kv_cache
    else:
        raise ValueError(f"Unsupported dtype: {kv_cache.dtype}")

    elements_per_layer_kv = n_heads * seq_len * head_dim
    if elements_per_layer_kv % 64 != 0:
        raise ValueError(
            f"Elements per layer/KV ({elements_per_layer_kv}) must be multiple of 64. "
            f"Pad seq_len or head_dim."
        )

    n_groups = elements_per_layer_kv // 64

    # Allocate output arrays
    # weights: 32 i32 per group (one i32 per packed byte)
    all_weights = np.empty((n_layers, 2, n_groups * 32), dtype=np.int32)
    all_scales = np.empty((n_layers, 2, n_groups), dtype=np.float32)
    all_biases = np.empty((n_layers, 2, n_groups), dtype=np.float32)

    for layer in range(n_layers):
        for kv_idx in range(2):
            flat = np.ascontiguousarray(kv_f32[layer, kv_idx].ravel())
            q4_quantize_f32(
                flat,
                all_weights[layer, kv_idx],
                all_scales[layer, kv_idx],
                all_biases[layer, kv_idx],
                n_groups,
            )

    return Q4Bundle(
        weights=all_weights,
        scales=all_scales,
        biases=all_biases,
        n_layers=n_layers,
        n_heads=n_heads,
        head_dim=head_dim,
        seq_len=seq_len,
        orig_dtype=orig_dtype,
    )
