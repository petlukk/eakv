"""Tests for .eakv binary file format and mmap support."""

import numpy as np
import pytest
from eakv._quantize import quantize
from eakv._restore import dequantize
from eakv._io import save, load, open_mmap


def _make_bundle():
    kv = np.random.default_rng(42).standard_normal((2, 2, 4, 64, 64)).astype(np.float32)
    return quantize(kv), kv


class TestSaveLoad:
    def test_roundtrip(self, tmp_path):
        bundle, _ = _make_bundle()
        path = str(tmp_path / "test.eakv")
        save(bundle, path)
        loaded = load(path)
        restored = dequantize(loaded)
        original = dequantize(bundle)
        np.testing.assert_array_equal(restored, original)

    def test_metadata_preserved(self, tmp_path):
        bundle, _ = _make_bundle()
        bundle.model_hash = "abc123"
        bundle.tokenizer_hash = "def456"
        path = str(tmp_path / "test.eakv")
        save(bundle, path)
        loaded = load(path)
        assert loaded.n_layers == bundle.n_layers
        assert loaded.n_heads == bundle.n_heads
        assert loaded.seq_len == bundle.seq_len
        assert loaded.head_dim == bundle.head_dim
        assert loaded.model_hash == "abc123"
        assert loaded.tokenizer_hash == "def456"

    def test_magic_bytes(self, tmp_path):
        bundle, _ = _make_bundle()
        path = str(tmp_path / "test.eakv")
        save(bundle, path)
        with open(path, "rb") as f:
            assert f.read(4) == b"EAKV"

    def test_bad_magic(self, tmp_path):
        path = str(tmp_path / "bad.eakv")
        with open(path, "wb") as f:
            f.write(b"NOPE" + b"\x00" * 508)
        with pytest.raises(ValueError, match="magic"):
            load(path)


class TestMmap:
    def test_mmap_restore(self, tmp_path):
        bundle, _ = _make_bundle()
        path = str(tmp_path / "test.eakv")
        save(bundle, path)
        with open_mmap(path) as mmapped:
            restored = dequantize(mmapped)
            original = dequantize(bundle)
            np.testing.assert_array_equal(restored, original)

    def test_mmap_metadata(self, tmp_path):
        bundle, _ = _make_bundle()
        path = str(tmp_path / "test.eakv")
        save(bundle, path)
        with open_mmap(path) as mmapped:
            assert mmapped.n_layers == bundle.n_layers
            assert mmapped.n_heads == bundle.n_heads
