"""Fused attention operations on packed Q4 KV cache."""

import ctypes as _ct
import numpy as np
from numpy.typing import NDArray
from pathlib import Path as _Path

from ._bundle import Q4Bundle

_LIB_DIR = _Path(__file__).parent / "lib"
_P_U8 = _ct.POINTER(_ct.c_uint8)
_P_F32 = _ct.POINTER(_ct.c_float)
_I32 = _ct.c_int32

# Load fused kernel libraries
_k_lib = _ct.CDLL(str(_LIB_DIR / "libfused_k_score.so"))
_k_lib.q4_fused_k_score_f32.argtypes = [
    _P_F32, _P_U8, _P_F32, _P_F32, _P_F32, _I32, _I32,
]
_k_lib.q4_fused_k_score_f32.restype = None

_k_lib.q4_fused_k_score_multi_f32.argtypes = [
    _P_F32, _P_U8, _P_F32, _P_F32, _P_F32, _I32, _I32, _I32,
]
_k_lib.q4_fused_k_score_multi_f32.restype = None

_v_lib = _ct.CDLL(str(_LIB_DIR / "libfused_v_sum.so"))
_v_lib.q4_fused_v_sum_f32.argtypes = [
    _P_F32, _P_U8, _P_F32, _P_F32, _P_F32, _I32, _I32,
]
_v_lib.q4_fused_v_sum_f32.restype = None

_v_lib.q4_fused_v_sum_multi_f32.argtypes = [
    _P_F32, _P_U8, _P_F32, _P_F32, _P_F32, _I32, _I32, _I32,
]
_v_lib.q4_fused_v_sum_multi_f32.restype = None

_a_lib = _ct.CDLL(str(_LIB_DIR / "libfused_attention.so"))
_a_lib.q4_fused_attention_f32.argtypes = [
    _P_F32, _P_U8, _P_F32, _P_F32, _P_U8, _P_F32, _P_F32, _P_F32, _I32, _I32,
]
_a_lib.q4_fused_attention_f32.restype = None

_a_lib.q4_fused_attention_multi_f32.argtypes = [
    _P_F32, _P_U8, _P_F32, _P_F32, _P_U8, _P_F32, _P_F32, _P_F32, _I32, _I32, _I32,
]
_a_lib.q4_fused_attention_multi_f32.restype = None

_gqa_lib = _ct.CDLL(str(_LIB_DIR / "libfused_k_score_gqa.so"))
_gqa_lib.q4_k_score_gqa_f32.argtypes = [
    _P_F32, _P_U8, _P_F32, _P_F32, _P_F32, _I32, _I32, _I32, _I32,
]
_gqa_lib.q4_k_score_gqa_f32.restype = None

_gqa_lib.q4_v_sum_gqa_f32.argtypes = [
    _P_F32, _P_U8, _P_F32, _P_F32, _P_F32, _I32, _I32, _I32, _I32,
]
_gqa_lib.q4_v_sum_gqa_f32.restype = None


def attention_scores(
    bundle: Q4Bundle,
    query: NDArray,
    layer: int,
    head: int = 0,
) -> NDArray:
    """Compute attention scores directly from packed Q4 K cache.

    Args:
        bundle: Q4Bundle with quantized KV cache
        query: shape [128] f32 — single query vector
        layer: transformer layer index
        head: KV head index

    Returns:
        scores: shape [seq_len] f32 — raw scores (pre-softmax, scaled by 1/sqrt(d))
    """
    if bundle.head_dim != 128:
        raise ValueError(f"Fused attention requires head_dim=128, got {bundle.head_dim}")
    if query.shape != (128,):
        raise ValueError(f"Query must be shape (128,), got {query.shape}")

    q_vec = np.ascontiguousarray(query, dtype=np.float32)
    scores = np.empty(bundle.seq_len, dtype=np.float32)
    group_offset = head * bundle.seq_len * 2

    _k_lib.q4_fused_k_score_f32(
        q_vec.ctypes.data_as(_P_F32),
        bundle.weights[layer, 0].ctypes.data_as(_P_U8),
        bundle.scales[layer, 0].ctypes.data_as(_P_F32),
        bundle.biases[layer, 0].ctypes.data_as(_P_F32),
        scores.ctypes.data_as(_P_F32),
        _I32(bundle.seq_len),
        _I32(group_offset),
    )
    return scores


def attention_output(
    bundle: Q4Bundle,
    weights: NDArray,
    layer: int,
    head: int = 0,
) -> NDArray:
    """Compute weighted V sum directly from packed Q4 V cache.

    Args:
        bundle: Q4Bundle with quantized KV cache
        weights: shape [seq_len] f32 — softmax attention weights
        layer: transformer layer index
        head: KV head index

    Returns:
        out: shape [128] f32 — attention output vector
    """
    if bundle.head_dim != 128:
        raise ValueError(f"Fused attention requires head_dim=128, got {bundle.head_dim}")

    w = np.ascontiguousarray(weights, dtype=np.float32)
    out_vec = np.zeros(128, dtype=np.float32)
    group_offset = head * bundle.seq_len * 2

    _v_lib.q4_fused_v_sum_f32(
        w.ctypes.data_as(_P_F32),
        bundle.weights[layer, 1].ctypes.data_as(_P_U8),
        bundle.scales[layer, 1].ctypes.data_as(_P_F32),
        bundle.biases[layer, 1].ctypes.data_as(_P_F32),
        out_vec.ctypes.data_as(_P_F32),
        _I32(bundle.seq_len),
        _I32(group_offset),
    )
    return out_vec


def attention_scores_multi(
    bundle: Q4Bundle,
    q_vecs: NDArray,
    layer: int,
    n_heads: int,
) -> NDArray:
    """Compute attention scores for all heads in one kernel call.

    Args:
        bundle: Q4Bundle with quantized KV cache
        q_vecs: shape [n_heads, 128] f32 — query vectors for all heads
        layer: transformer layer index
        n_heads: number of heads to process

    Returns:
        scores: shape [n_heads, seq_len] f32 — raw scores (pre-softmax)
    """
    if bundle.head_dim != 128:
        raise ValueError(f"Fused attention requires head_dim=128, got {bundle.head_dim}")

    q_flat = np.ascontiguousarray(q_vecs.reshape(-1), dtype=np.float32)
    scores = np.empty(n_heads * bundle.seq_len, dtype=np.float32)
    groups_per_head = bundle.seq_len * 2

    _k_lib.q4_fused_k_score_multi_f32(
        q_flat.ctypes.data_as(_P_F32),
        bundle.weights[layer, 0].ctypes.data_as(_P_U8),
        bundle.scales[layer, 0].ctypes.data_as(_P_F32),
        bundle.biases[layer, 0].ctypes.data_as(_P_F32),
        scores.ctypes.data_as(_P_F32),
        _I32(bundle.seq_len),
        _I32(n_heads),
        _I32(groups_per_head),
    )
    return scores.reshape(n_heads, bundle.seq_len)


def attention_output_multi(
    bundle: Q4Bundle,
    all_weights: NDArray,
    layer: int,
    n_heads: int,
) -> NDArray:
    """Compute weighted V sum for all heads in one kernel call.

    Args:
        bundle: Q4Bundle with quantized KV cache
        all_weights: shape [n_heads, seq_len] f32 — softmax weights for all heads
        layer: transformer layer index
        n_heads: number of heads to process

    Returns:
        out: shape [n_heads, 128] f32 — attention output vectors
    """
    if bundle.head_dim != 128:
        raise ValueError(f"Fused attention requires head_dim=128, got {bundle.head_dim}")

    w_flat = np.ascontiguousarray(all_weights.reshape(-1), dtype=np.float32)
    out = np.empty(n_heads * 128, dtype=np.float32)
    groups_per_head = bundle.seq_len * 2

    _v_lib.q4_fused_v_sum_multi_f32(
        w_flat.ctypes.data_as(_P_F32),
        bundle.weights[layer, 1].ctypes.data_as(_P_U8),
        bundle.scales[layer, 1].ctypes.data_as(_P_F32),
        bundle.biases[layer, 1].ctypes.data_as(_P_F32),
        out.ctypes.data_as(_P_F32),
        _I32(bundle.seq_len),
        _I32(n_heads),
        _I32(groups_per_head),
    )
    return out.reshape(n_heads, 128)


def attention_scores_gqa(
    bundle: Q4Bundle,
    q_vecs: NDArray,
    layer: int,
    n_q_heads: int,
    n_kv_heads: int,
) -> NDArray:
    """Compute K-scores with GQA loop flip: K dequantized once, reused across Q heads.

    Q heads [kv_h * q_per_kv .. (kv_h+1) * q_per_kv) share KV head kv_h.
    For MHA (n_q_heads == n_kv_heads), equivalent to attention_scores_multi.

    Args:
        bundle: Q4Bundle with quantized KV cache (n_kv_heads heads)
        q_vecs: shape [n_q_heads, 128] f32 — query vectors
        layer: transformer layer index
        n_q_heads: total query heads
        n_kv_heads: KV heads (must divide n_q_heads evenly)

    Returns:
        scores: shape [n_q_heads, seq_len] f32
    """
    if bundle.head_dim != 128:
        raise ValueError(f"Fused attention requires head_dim=128, got {bundle.head_dim}")

    q_flat = np.ascontiguousarray(q_vecs.reshape(-1), dtype=np.float32)
    scores = np.empty(n_q_heads * bundle.seq_len, dtype=np.float32)
    groups_per_head = bundle.seq_len * 2

    _gqa_lib.q4_k_score_gqa_f32(
        q_flat.ctypes.data_as(_P_F32),
        bundle.weights[layer, 0].ctypes.data_as(_P_U8),
        bundle.scales[layer, 0].ctypes.data_as(_P_F32),
        bundle.biases[layer, 0].ctypes.data_as(_P_F32),
        scores.ctypes.data_as(_P_F32),
        _I32(bundle.seq_len),
        _I32(n_q_heads),
        _I32(n_kv_heads),
        _I32(groups_per_head),
    )
    return scores.reshape(n_q_heads, bundle.seq_len)


def attention_output_gqa(
    bundle: Q4Bundle,
    all_weights: NDArray,
    layer: int,
    n_q_heads: int,
    n_kv_heads: int,
) -> NDArray:
    """Compute V-sum with GQA: V dequantized once, paired Q heads accumulated together.

    Args:
        bundle: Q4Bundle with quantized KV cache (n_kv_heads heads)
        all_weights: shape [n_q_heads, seq_len] f32 — softmax weights
        layer: transformer layer index
        n_q_heads: total query heads
        n_kv_heads: KV heads (must divide n_q_heads evenly)

    Returns:
        out: shape [n_q_heads, 128] f32
    """
    if bundle.head_dim != 128:
        raise ValueError(f"Fused attention requires head_dim=128, got {bundle.head_dim}")

    w_flat = np.ascontiguousarray(all_weights.reshape(-1), dtype=np.float32)
    out = np.empty(n_q_heads * 128, dtype=np.float32)
    groups_per_head = bundle.seq_len * 2

    _gqa_lib.q4_v_sum_gqa_f32(
        w_flat.ctypes.data_as(_P_F32),
        bundle.weights[layer, 1].ctypes.data_as(_P_U8),
        bundle.scales[layer, 1].ctypes.data_as(_P_F32),
        bundle.biases[layer, 1].ctypes.data_as(_P_F32),
        out.ctypes.data_as(_P_F32),
        _I32(bundle.seq_len),
        _I32(n_q_heads),
        _I32(n_kv_heads),
        _I32(groups_per_head),
    )
    return out.reshape(n_q_heads, 128)


def attention_fused(
    bundle: Q4Bundle,
    query: NDArray,
    layer: int,
    head: int = 0,
) -> NDArray:
    """Compute softmax(Q·K^T/sqrt(d))·V in one fused kernel call.

    No intermediate score or weight arrays are allocated.

    Args:
        bundle: Q4Bundle with quantized KV cache
        query: shape [128] f32 — single query vector
        layer: transformer layer index
        head: KV head index

    Returns:
        out: shape [128] f32 — attention output vector
    """
    if bundle.head_dim != 128:
        raise ValueError(f"Fused attention requires head_dim=128, got {bundle.head_dim}")
    if query.shape != (128,):
        raise ValueError(f"Query must be shape (128,), got {query.shape}")

    q_vec = np.ascontiguousarray(query, dtype=np.float32)
    out_vec = np.empty(128, dtype=np.float32)
    group_offset = head * bundle.seq_len * 2

    _a_lib.q4_fused_attention_f32(
        q_vec.ctypes.data_as(_P_F32),
        bundle.weights[layer, 0].ctypes.data_as(_P_U8),
        bundle.scales[layer, 0].ctypes.data_as(_P_F32),
        bundle.biases[layer, 0].ctypes.data_as(_P_F32),
        bundle.weights[layer, 1].ctypes.data_as(_P_U8),
        bundle.scales[layer, 1].ctypes.data_as(_P_F32),
        bundle.biases[layer, 1].ctypes.data_as(_P_F32),
        out_vec.ctypes.data_as(_P_F32),
        _I32(bundle.seq_len),
        _I32(group_offset),
    )
    return out_vec


def attention_fused_multi(
    bundle: Q4Bundle,
    q_vecs: NDArray,
    layer: int,
    n_heads: int,
) -> NDArray:
    """Compute softmax(Q·K^T/sqrt(d))·V for all heads in one kernel call.

    Args:
        bundle: Q4Bundle with quantized KV cache
        q_vecs: shape [n_heads, 128] f32 — query vectors
        layer: transformer layer index
        n_heads: number of heads to process

    Returns:
        out: shape [n_heads, 128] f32 — attention output vectors
    """
    if bundle.head_dim != 128:
        raise ValueError(f"Fused attention requires head_dim=128, got {bundle.head_dim}")

    q_flat = np.ascontiguousarray(q_vecs.reshape(-1), dtype=np.float32)
    out = np.empty(n_heads * 128, dtype=np.float32)
    groups_per_head = bundle.seq_len * 2

    _a_lib.q4_fused_attention_multi_f32(
        q_flat.ctypes.data_as(_P_F32),
        bundle.weights[layer, 0].ctypes.data_as(_P_U8),
        bundle.scales[layer, 0].ctypes.data_as(_P_F32),
        bundle.biases[layer, 0].ctypes.data_as(_P_F32),
        bundle.weights[layer, 1].ctypes.data_as(_P_U8),
        bundle.scales[layer, 1].ctypes.data_as(_P_F32),
        bundle.biases[layer, 1].ctypes.data_as(_P_F32),
        out.ctypes.data_as(_P_F32),
        _I32(bundle.seq_len),
        _I32(n_heads),
        _I32(groups_per_head),
    )
    return out.reshape(n_heads, 128)
