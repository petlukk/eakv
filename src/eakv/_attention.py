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

_v_lib = _ct.CDLL(str(_LIB_DIR / "libfused_v_sum.so"))
_v_lib.q4_fused_v_sum_f32.argtypes = [
    _P_F32, _P_U8, _P_F32, _P_F32, _P_F32, _I32, _I32,
]
_v_lib.q4_fused_v_sum_f32.restype = None


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
