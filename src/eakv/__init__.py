"""eakv: Fast Q4 KV cache quantization and restore for LLM inference."""

from ._bundle import Q4Bundle
from ._quantize import quantize
from ._restore import dequantize, restore
from ._io import save, load, open_mmap
from ._ops import q4_validate
from ._dispatch import get_isa
from ._attention import (
    attention_scores, attention_output,
    attention_scores_multi, attention_output_multi,
)

__version__ = "0.1.0"

__all__ = [
    "Q4Bundle",
    "quantize",
    "dequantize",
    "restore",
    "save",
    "load",
    "open_mmap",
    "validate",
    "attention_scores",
    "attention_output",
    "attention_scores_multi",
    "attention_output_multi",
]


def validate(bundle: Q4Bundle) -> None:
    """Validate a Q4Bundle. Raises ValueError on corruption."""
    import numpy as np
    for layer in range(bundle.n_layers):
        for kv_idx in range(2):
            scales = bundle.scales[layer, kv_idx]
            biases = bundle.biases[layer, kv_idx]
            scales_bits = np.ascontiguousarray(scales).view(np.int32)
            biases_bits = np.ascontiguousarray(biases).view(np.int32)
            result = q4_validate(
                scales,
                biases,
                scales_bits,
                biases_bits,
                len(scales),
            )
            kv_name = "K" if kv_idx == 0 else "V"
            if result == 1:
                raise ValueError(f"NaN scale in layer {layer} {kv_name}")
            elif result == 2:
                raise ValueError(f"NaN bias in layer {layer} {kv_name}")
            elif result == 3:
                raise ValueError(f"Negative scale in layer {layer} {kv_name}")
            elif result != 0:
                raise ValueError(f"Validation error {result} in layer {layer} {kv_name}")
