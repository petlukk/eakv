"""Q4Bundle: container for quantized KV cache data."""

from dataclasses import dataclass
from typing import Optional
import numpy as np
from numpy.typing import NDArray


@dataclass
class Q4Bundle:
    """Quantized KV cache in Q4_1 format.

    Attributes:
        weights: Packed values as u8, shape (n_layers, 2, n_groups_per_layer * 32)
                 Axis 1: 0=K, 1=V. Each byte is a packed pair of 4-bit values (0-255).
        scales: Per-group scales, shape (n_layers, 2, n_groups_per_layer) -- f32
        biases: Per-group biases, shape (n_layers, 2, n_groups_per_layer) -- f32
        n_layers: Number of transformer layers
        n_heads: Number of KV heads
        head_dim: Dimension per head
        seq_len: Current sequence length
        orig_dtype: Original dtype string ("float32" or "float16")
        model_hash: Optional model identity for safety
        tokenizer_hash: Optional tokenizer identity for safety
    """
    weights: NDArray
    scales: NDArray
    biases: NDArray
    n_layers: int
    n_heads: int
    head_dim: int
    seq_len: int
    orig_dtype: str = "float32"
    model_hash: Optional[str] = None
    tokenizer_hash: Optional[str] = None

    @property
    def n_groups_per_layer(self) -> int:
        return (self.n_heads * self.head_dim * self.seq_len) // 64

    @property
    def compressed_size(self) -> int:
        return self.weights.nbytes + self.scales.nbytes + self.biases.nbytes

    @property
    def original_size(self) -> int:
        elem_size = 4 if self.orig_dtype == "float32" else 2
        return self.n_layers * 2 * self.n_heads * self.head_dim * self.seq_len * elem_size

    @property
    def compression_ratio(self) -> float:
        return self.compressed_size / self.original_size
