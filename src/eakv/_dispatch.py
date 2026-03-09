"""Runtime ISA dispatch for SIMD dequantize kernels.

Detects CPU features and loads the fastest available kernel:
  AVX-512 > AVX2 > SSE (fallback)
"""

import ctypes as _ct
import numpy as _np
from pathlib import Path as _Path

_LIB_DIR = _Path(__file__).parent / "lib"

_P_U8 = _ct.POINTER(_ct.c_uint8)
_P_F32 = _ct.POINTER(_ct.c_float)
_I32 = _ct.c_int32


def _detect_isa():
    """Detect best available ISA using /proc/cpuinfo."""
    try:
        with open("/proc/cpuinfo", "r") as f:
            cpuinfo = f.read()
        flags = set()
        for line in cpuinfo.split("\n"):
            if line.startswith("flags"):
                flags = set(line.split(":")[1].split())
                break

        if "avx512f" in flags and "avx512bw" in flags and "avx512vl" in flags:
            return "avx512"
        elif "avx2" in flags and "fma" in flags:
            return "avx2"
        else:
            return "sse"
    except (FileNotFoundError, IndexError):
        return "sse"


_ISA = _detect_isa()

_LIB_MAP = {
    "avx512": ("libdequantize_avx512.so",
                "q4_dequantize_avx512_f32",
                "q4_dequantize_range_avx512_f32"),
    "avx2": ("libdequantize_avx2.so",
             "q4_dequantize_avx2_f32",
             "q4_dequantize_range_avx2_f32"),
    "sse": ("libdequantize_sse.so",
            "q4_dequantize_simd_f32",
            "q4_dequantize_range_simd_f32"),
}


def _load_kernel(isa):
    """Load dequantize library for given ISA."""
    lib_file, dq_name, rng_name = _LIB_MAP[isa]
    lib = _ct.CDLL(str(_LIB_DIR / lib_file))

    dq_fn = getattr(lib, dq_name)
    dq_fn.argtypes = [_P_U8, _P_F32, _P_F32, _P_F32, _I32]
    dq_fn.restype = None

    rng_fn = getattr(lib, rng_name)
    rng_fn.argtypes = [_P_U8, _P_F32, _P_F32, _P_F32, _I32, _I32]
    rng_fn.restype = None

    return dq_fn, rng_fn


_dequant_fn, _range_fn = _load_kernel(_ISA)


def get_isa():
    """Return the detected ISA tier: 'avx512', 'avx2', or 'sse'."""
    return _ISA


def q4_dequantize_dispatch(
    weights: _np.ndarray,
    scales: _np.ndarray,
    biases: _np.ndarray,
    out: _np.ndarray,
    n_groups: int,
):
    """Dequantize using the best available ISA."""
    _dequant_fn(
        weights.ctypes.data_as(_P_U8),
        scales.ctypes.data_as(_P_F32),
        biases.ctypes.data_as(_P_F32),
        out.ctypes.data_as(_P_F32),
        _I32(int(n_groups)),
    )


def q4_dequantize_range_dispatch(
    weights: _np.ndarray,
    scales: _np.ndarray,
    biases: _np.ndarray,
    out: _np.ndarray,
    group_start: int,
    group_count: int,
):
    """Partial dequantize using the best available ISA."""
    _range_fn(
        weights.ctypes.data_as(_P_U8),
        scales.ctypes.data_as(_P_F32),
        biases.ctypes.data_as(_P_F32),
        out.ctypes.data_as(_P_F32),
        _I32(int(group_start)),
        _I32(int(group_count)),
    )
