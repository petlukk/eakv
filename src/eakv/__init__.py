"""eakv: Fast Q4 KV cache quantization and restore for LLM inference."""

from ._bundle import Q4Bundle
from ._quantize import quantize
from ._restore import dequantize, restore
from ._io import save, load, open_mmap
from ._ops import q4_validate

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
]


def validate(bundle: Q4Bundle) -> None:
    """Validate a Q4Bundle. Raises ValueError on corruption."""
    for layer in range(bundle.n_layers):
        for kv_idx in range(2):
            result = q4_validate(
                bundle.scales[layer, kv_idx],
                bundle.biases[layer, kv_idx],
                len(bundle.scales[layer, kv_idx]),
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
