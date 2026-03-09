"""High-level restore API with partial restore support."""

from typing import Optional, Union
import numpy as np
from numpy.typing import NDArray

from ._bundle import Q4Bundle
from ._dispatch import q4_dequantize_dispatch, q4_dequantize_range_dispatch


def dequantize(bundle: Q4Bundle) -> NDArray:
    """Full dequantize: restore entire KV cache."""
    return restore(bundle)


def restore(
    bundle: Q4Bundle,
    layers: Optional[Union[range, list, int]] = None,
    tokens: Optional[Union[int, range, slice]] = None,
    heads: Optional[Union[range, list, int]] = None,
    out_dtype: str = "float32",
) -> NDArray:
    """Restore KV cache with optional partial selection."""
    # Resolve layer indices
    if layers is None:
        layer_indices = list(range(bundle.n_layers))
    elif isinstance(layers, int):
        layer_indices = [layers]
    else:
        layer_indices = list(layers)

    # Resolve head indices
    if heads is None:
        head_indices = list(range(bundle.n_heads))
    elif isinstance(heads, int):
        head_indices = [heads]
    else:
        head_indices = list(heads)

    # Resolve token range
    if tokens is None:
        token_start, token_count = 0, bundle.seq_len
    elif isinstance(tokens, int):
        if tokens < 0:
            token_start = bundle.seq_len + tokens
            token_count = -tokens
        else:
            token_start = 0
            token_count = tokens
    elif isinstance(tokens, (range, slice)):
        start = tokens.start or 0
        stop = tokens.stop or bundle.seq_len
        if start < 0:
            start = bundle.seq_len + start
        if stop < 0:
            stop = bundle.seq_len + stop
        token_start = start
        token_count = stop - start
    else:
        raise ValueError(f"Unsupported tokens type: {type(tokens)}")

    # Full restore fast path
    if (len(layer_indices) == bundle.n_layers
            and len(head_indices) == bundle.n_heads
            and token_start == 0 and token_count == bundle.seq_len):
        return _full_restore(bundle, out_dtype)

    # Partial restore
    n_out_layers = len(layer_indices)
    n_out_heads = len(head_indices)

    out = np.empty(
        (n_out_layers, 2, n_out_heads, token_count, bundle.head_dim),
        dtype=np.float32,
    )

    _dequant = _pick_dequantize(bundle)

    for li, layer in enumerate(layer_indices):
        for kv_idx in range(2):
            # Full dequantize this layer/kv
            n_groups = bundle.n_groups_per_layer
            full_flat = np.empty(n_groups * 64, dtype=np.float32)
            _dequant(
                bundle.weights[layer, kv_idx],
                bundle.scales[layer, kv_idx],
                bundle.biases[layer, kv_idx],
                full_flat,
                n_groups,
            )
            full = full_flat.reshape(bundle.n_heads, bundle.seq_len, bundle.head_dim)

            for hi, head in enumerate(head_indices):
                out[li, kv_idx, hi] = full[head, token_start:token_start + token_count]

    if out_dtype == "float16":
        out = out.astype(np.float16)

    return out


def _pick_dequantize(bundle: Q4Bundle):
    """Select the dispatched SIMD dequantize kernel."""
    return q4_dequantize_dispatch


def _full_restore(bundle: Q4Bundle, out_dtype: str) -> NDArray:
    out = np.empty(
        (bundle.n_layers, 2, bundle.n_heads, bundle.seq_len, bundle.head_dim),
        dtype=np.float32,
    )
    n_groups = bundle.n_groups_per_layer
    _dequant = _pick_dequantize(bundle)

    for layer in range(bundle.n_layers):
        for kv_idx in range(2):
            flat = np.empty(n_groups * 64, dtype=np.float32)
            _dequant(
                bundle.weights[layer, kv_idx],
                bundle.scales[layer, kv_idx],
                bundle.biases[layer, kv_idx],
                flat,
                n_groups,
            )
            out[layer, kv_idx] = flat.reshape(bundle.n_heads, bundle.seq_len, bundle.head_dim)

    if out_dtype == "float16":
        out = out.astype(np.float16)

    return out
