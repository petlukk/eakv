"""Test Q4_1 quantize/dequantize roundtrip accuracy."""

import numpy as np
import pytest

from eakv._quantize import quantize
from eakv._restore import dequantize, restore


def _make_kv_cache(n_layers=2, n_heads=4, seq_len=64, head_dim=64, seed=42):
    rng = np.random.default_rng(seed)
    return rng.standard_normal((n_layers, 2, n_heads, seq_len, head_dim)).astype(np.float32)


class TestRoundtrip:
    def test_basic_roundtrip(self):
        kv = _make_kv_cache()
        bundle = quantize(kv)
        restored = dequantize(bundle)
        assert restored.shape == kv.shape
        max_err = np.max(np.abs(kv - restored))
        assert max_err < 0.5, f"Max error too large: {max_err}"

    def test_bundle_metadata(self):
        kv = _make_kv_cache(n_layers=4, n_heads=8, seq_len=128, head_dim=128)
        bundle = quantize(kv)
        assert bundle.n_layers == 4
        assert bundle.n_heads == 8
        assert bundle.seq_len == 128
        assert bundle.head_dim == 128
        assert bundle.orig_dtype == "float32"

    def test_uniform_values(self):
        kv = np.full((1, 2, 1, 64, 64), 3.14, dtype=np.float32)
        bundle = quantize(kv)
        restored = dequantize(bundle)
        np.testing.assert_allclose(restored, kv, atol=1e-5)

    def test_zero_values(self):
        kv = np.zeros((1, 2, 1, 64, 64), dtype=np.float32)
        bundle = quantize(kv)
        restored = dequantize(bundle)
        np.testing.assert_allclose(restored, kv, atol=1e-6)

    @pytest.mark.parametrize("seq_len", [64, 128, 256, 512])
    def test_various_seq_lengths(self, seq_len):
        kv = _make_kv_cache(seq_len=seq_len)
        bundle = quantize(kv)
        restored = dequantize(bundle)
        assert restored.shape == kv.shape
        max_err = np.max(np.abs(kv - restored))
        assert max_err < 0.5


class TestPartialRestore:
    def test_layer_selection(self):
        kv = _make_kv_cache(n_layers=4)
        bundle = quantize(kv)
        partial = restore(bundle, layers=[0, 2])
        assert partial.shape[0] == 2
        full = dequantize(bundle)
        np.testing.assert_array_equal(partial[0], full[0])
        np.testing.assert_array_equal(partial[1], full[2])

    def test_last_n_tokens(self):
        kv = _make_kv_cache(seq_len=256)
        bundle = quantize(kv)
        partial = restore(bundle, tokens=-64)
        assert partial.shape[3] == 64
        full = dequantize(bundle)
        np.testing.assert_array_equal(partial, full[:, :, :, -64:, :])

    def test_head_selection(self):
        kv = _make_kv_cache(n_heads=8)
        bundle = quantize(kv)
        partial = restore(bundle, heads=[0, 3, 7])
        assert partial.shape[2] == 3
        full = dequantize(bundle)
        np.testing.assert_array_equal(partial[:, :, 0], full[:, :, 0])
        np.testing.assert_array_equal(partial[:, :, 1], full[:, :, 3])
        np.testing.assert_array_equal(partial[:, :, 2], full[:, :, 7])

    def test_combined_partial(self):
        kv = _make_kv_cache(n_layers=4, n_heads=8, seq_len=256)
        bundle = quantize(kv)
        partial = restore(bundle, layers=[1], tokens=-128, heads=[0, 4])
        assert partial.shape == (1, 2, 2, 128, 64)

    def test_out_dtype_f16(self):
        kv = _make_kv_cache()
        bundle = quantize(kv)
        restored = restore(bundle, out_dtype="float16")
        assert restored.dtype == np.float16


class TestInputValidation:
    def test_wrong_ndim(self):
        with pytest.raises(ValueError, match="5D"):
            quantize(np.zeros((2, 3), dtype=np.float32))

    def test_wrong_kv_dim(self):
        with pytest.raises(ValueError, match="axis 1"):
            quantize(np.zeros((2, 3, 4, 64, 64), dtype=np.float32))

    def test_not_multiple_of_64(self):
        with pytest.raises(ValueError, match="multiple of 64"):
            quantize(np.zeros((1, 2, 1, 63, 1), dtype=np.float32))
