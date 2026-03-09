"""Binary .eakv file format with mmap support."""

import mmap
import struct
from contextlib import contextmanager
from typing import Optional

import numpy as np

from ._bundle import Q4Bundle

MAGIC = b"EAKV"
HEADER_SIZE = 512
ALIGNMENT = 64

# After magic (4 bytes):
# version(u16) quant_scheme(u16) group_size(u32) orig_dtype(u16)
# n_layers(u32) n_heads(u32) head_dim(u32) seq_len(u32) max_seq_len(u32)
# compression(i16) model_hash(32s) tokenizer_hash(32s) checksum(u64)
HEADER_STRUCT = struct.Struct("<HHIHIIIIIh32s32sQ")

_DTYPE_MAP = {"float32": 0, "float16": 1}
_DTYPE_RMAP = {0: "float32", 1: "float16"}


def _align64(offset: int) -> int:
    """Round up to next 64-byte boundary."""
    return (offset + 63) & ~63


def _encode_hash(h: Optional[str], size: int = 32) -> bytes:
    if h is None:
        return b"\x00" * size
    encoded = h.encode("utf-8")[:size]
    return encoded.ljust(size, b"\x00")


def _decode_hash(raw: bytes) -> Optional[str]:
    stripped = raw.rstrip(b"\x00")
    if not stripped:
        return None
    return stripped.decode("utf-8")


def save(
    bundle: Q4Bundle,
    path: str,
    compress: Optional[str] = None,
    model_hash: Optional[str] = None,
    tokenizer_hash: Optional[str] = None,
) -> None:
    """Save Q4Bundle to .eakv file."""
    if compress is not None:
        raise NotImplementedError("Compression not yet supported")

    n_layers = bundle.n_layers
    mh = model_hash if model_hash is not None else bundle.model_hash
    th = tokenizer_hash if tokenizer_hash is not None else bundle.tokenizer_hash

    # Build header
    header_fields = HEADER_STRUCT.pack(
        1,  # version
        0,  # quant_scheme (Q4_1)
        64,  # group_size
        _DTYPE_MAP.get(bundle.orig_dtype, 0),
        n_layers,
        bundle.n_heads,
        bundle.head_dim,
        bundle.seq_len,
        bundle.seq_len,  # max_seq_len = seq_len
        0,  # compression = none
        _encode_hash(mh),
        _encode_hash(th),
        0,  # checksum reserved
    )

    header = MAGIC + header_fields
    header = header.ljust(HEADER_SIZE, b"\x00")

    n_groups = bundle.n_groups_per_layer

    # Pre-compute layer data and offsets
    index_table_size = n_layers * 2 * 8  # K_offset, V_offset per layer as u64
    data_start = HEADER_SIZE + index_table_size
    data_start = _align64(data_start)

    # Weights per layer/kv: n_groups * 32 uint8 bytes
    weights_size = n_groups * 32
    scales_size = n_groups * 4  # f32
    biases_size = n_groups * 4  # f32
    block_raw = weights_size + scales_size + biases_size
    block_aligned = _align64(block_raw)

    # Compute offsets
    offsets = []  # list of (k_offset, v_offset)
    cur = data_start
    for _ in range(n_layers):
        k_off = cur
        cur += block_aligned
        v_off = cur
        cur += block_aligned
        offsets.append((k_off, v_off))

    with open(path, "wb") as f:
        # Write header
        f.write(header)

        # Write index table
        for k_off, v_off in offsets:
            f.write(struct.pack("<QQ", k_off, v_off))

        # Pad to data_start
        cur_pos = HEADER_SIZE + index_table_size
        if cur_pos < data_start:
            f.write(b"\x00" * (data_start - cur_pos))

        # Write data blocks
        for layer in range(n_layers):
            for kv_idx in range(2):
                w = np.ascontiguousarray(bundle.weights[layer, kv_idx])
                if w.dtype != np.uint8:
                    w = w.astype(np.uint8)
                s = np.ascontiguousarray(bundle.scales[layer, kv_idx])
                b = np.ascontiguousarray(bundle.biases[layer, kv_idx])

                f.write(w.tobytes())
                f.write(s.tobytes())
                f.write(b.tobytes())

                # Pad to alignment
                written = weights_size + scales_size + biases_size
                pad = block_aligned - written
                if pad > 0:
                    f.write(b"\x00" * pad)


def load(path: str) -> Q4Bundle:
    """Load Q4Bundle from .eakv file."""
    with open(path, "rb") as f:
        data = f.read()

    return _parse(data)


def _parse(data, source_mmap=None):
    """Parse .eakv data (bytes or memoryview) into Q4Bundle."""
    magic = data[:4]
    if magic != MAGIC:
        raise ValueError(f"Bad magic: expected {MAGIC!r}, got {magic!r}")

    fields = HEADER_STRUCT.unpack_from(data, 4)
    (version, quant_scheme, group_size, orig_dtype_code,
     n_layers, n_heads, head_dim, seq_len, max_seq_len,
     compression, model_hash_raw, tokenizer_hash_raw, checksum) = fields

    if version != 1:
        raise ValueError(f"Unsupported version: {version}")
    if compression != 0:
        raise NotImplementedError("Compressed files not yet supported")

    orig_dtype = _DTYPE_RMAP.get(orig_dtype_code, "float32")
    model_hash = _decode_hash(model_hash_raw)
    tokenizer_hash = _decode_hash(tokenizer_hash_raw)

    n_groups = (n_heads * head_dim * seq_len) // group_size

    # Read index table
    idx_offset = HEADER_SIZE
    offsets = []
    for i in range(n_layers):
        k_off, v_off = struct.unpack_from("<QQ", data, idx_offset + i * 16)
        offsets.append((k_off, v_off))

    weights_size = n_groups * 32
    scales_size = n_groups * 4
    biases_size = n_groups * 4

    all_weights = np.empty((n_layers, 2, n_groups * 32), dtype=np.uint8)
    all_scales = np.empty((n_layers, 2, n_groups), dtype=np.float32)
    all_biases = np.empty((n_layers, 2, n_groups), dtype=np.float32)

    for layer in range(n_layers):
        for kv_idx in range(2):
            off = offsets[layer][kv_idx]

            w_bytes = data[off:off + weights_size]
            all_weights[layer, kv_idx] = np.frombuffer(w_bytes, dtype=np.uint8).copy()

            s_off = off + weights_size
            s_bytes = data[s_off:s_off + scales_size]
            all_scales[layer, kv_idx] = np.frombuffer(s_bytes, dtype=np.float32).copy()

            b_off = s_off + scales_size
            b_bytes = data[b_off:b_off + biases_size]
            all_biases[layer, kv_idx] = np.frombuffer(b_bytes, dtype=np.float32).copy()

    return Q4Bundle(
        weights=all_weights,
        scales=all_scales,
        biases=all_biases,
        n_layers=n_layers,
        n_heads=n_heads,
        head_dim=head_dim,
        seq_len=seq_len,
        orig_dtype=orig_dtype,
        model_hash=model_hash,
        tokenizer_hash=tokenizer_hash,
    )


@contextmanager
def open_mmap(path: str):
    """Memory-map a .eakv file. Yields Q4Bundle backed by mmap.

    The arrays are copies from the mmap for safe use after close.
    """
    f = open(path, "rb")
    try:
        mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        try:
            bundle = _parse_mmap(mm)
            yield bundle
        finally:
            mm.close()
    finally:
        f.close()


def _parse_mmap(mm):
    """Parse mmap'd data into Q4Bundle with mmap-backed arrays."""
    magic = mm[:4]
    if magic != MAGIC:
        raise ValueError(f"Bad magic: expected {MAGIC!r}, got {magic!r}")

    fields = HEADER_STRUCT.unpack_from(mm, 4)
    (version, quant_scheme, group_size, orig_dtype_code,
     n_layers, n_heads, head_dim, seq_len, max_seq_len,
     compression, model_hash_raw, tokenizer_hash_raw, checksum) = fields

    if version != 1:
        raise ValueError(f"Unsupported version: {version}")
    if compression != 0:
        raise NotImplementedError("Compressed files not yet supported")

    orig_dtype = _DTYPE_RMAP.get(orig_dtype_code, "float32")
    model_hash = _decode_hash(model_hash_raw)
    tokenizer_hash = _decode_hash(tokenizer_hash_raw)

    n_groups = (n_heads * head_dim * seq_len) // group_size

    # Read index table
    idx_offset = HEADER_SIZE
    offsets = []
    for i in range(n_layers):
        k_off, v_off = struct.unpack_from("<QQ", mm, idx_offset + i * 16)
        offsets.append((k_off, v_off))

    weights_size = n_groups * 32
    scales_size = n_groups * 4
    biases_size = n_groups * 4

    all_weights = np.empty((n_layers, 2, n_groups * 32), dtype=np.uint8)
    all_scales = np.empty((n_layers, 2, n_groups), dtype=np.float32)
    all_biases = np.empty((n_layers, 2, n_groups), dtype=np.float32)

    for layer in range(n_layers):
        for kv_idx in range(2):
            off = offsets[layer][kv_idx]

            all_weights[layer, kv_idx] = np.frombuffer(
                mm, dtype=np.uint8, count=weights_size, offset=off
            ).copy()

            s_off = off + weights_size
            all_scales[layer, kv_idx] = np.frombuffer(
                mm, dtype=np.float32, count=n_groups, offset=s_off
            ).copy()

            b_off = s_off + scales_size
            all_biases[layer, kv_idx] = np.frombuffer(
                mm, dtype=np.float32, count=n_groups, offset=b_off
            ).copy()

    return Q4Bundle(
        weights=all_weights,
        scales=all_scales,
        biases=all_biases,
        n_layers=n_layers,
        n_heads=n_heads,
        head_dim=head_dim,
        seq_len=seq_len,
        orig_dtype=orig_dtype,
        model_hash=model_hash,
        tokenizer_hash=tokenizer_hash,
    )
