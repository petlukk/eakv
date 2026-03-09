"""Test fused attention kernels against reference restore-then-compute."""

import numpy as np
import pytest

from eakv._quantize import quantize
from eakv._restore import dequantize
from eakv._attention import attention_scores, attention_output


def _make_kv_cache(n_layers=1, n_heads=4, seq_len=64, head_dim=128, seed=42):
    rng = np.random.default_rng(seed)
    return rng.standard_normal((n_layers, 2, n_heads, seq_len, head_dim)).astype(np.float32)


def _softmax(x):
    e = np.exp(x - np.max(x))
    return e / e.sum()


class TestFusedKScore:
    def test_basic_correctness(self):
        kv = _make_kv_cache()
        bundle = quantize(kv)
        restored = dequantize(bundle)
        q_vec = np.random.default_rng(99).standard_normal(128).astype(np.float32)
        ref_scores = restored[0, 0, 0] @ q_vec / np.sqrt(128.0)
        fused_scores = attention_scores(bundle, q_vec, layer=0, head=0)
        np.testing.assert_allclose(fused_scores, ref_scores, atol=1e-4)

    def test_all_heads(self):
        kv = _make_kv_cache(n_heads=4)
        bundle = quantize(kv)
        restored = dequantize(bundle)
        q_vec = np.random.default_rng(99).standard_normal(128).astype(np.float32)
        for head in range(4):
            ref = restored[0, 0, head] @ q_vec / np.sqrt(128.0)
            fused = attention_scores(bundle, q_vec, layer=0, head=head)
            np.testing.assert_allclose(fused, ref, atol=1e-4, err_msg=f"head {head}")

    def test_longer_sequence(self):
        kv = _make_kv_cache(seq_len=512)
        bundle = quantize(kv)
        restored = dequantize(bundle)
        q_vec = np.random.default_rng(99).standard_normal(128).astype(np.float32)
        ref = restored[0, 0, 0] @ q_vec / np.sqrt(128.0)
        fused = attention_scores(bundle, q_vec, layer=0, head=0)
        np.testing.assert_allclose(fused, ref, atol=1e-4)

    def test_multiple_layers(self):
        kv = _make_kv_cache(n_layers=4)
        bundle = quantize(kv)
        restored = dequantize(bundle)
        q_vec = np.random.default_rng(99).standard_normal(128).astype(np.float32)
        for layer in range(4):
            ref = restored[layer, 0, 0] @ q_vec / np.sqrt(128.0)
            fused = attention_scores(bundle, q_vec, layer=layer, head=0)
            np.testing.assert_allclose(fused, ref, atol=1e-4, err_msg=f"layer {layer}")


class TestFusedVSum:
    def test_basic_correctness(self):
        kv = _make_kv_cache()
        bundle = quantize(kv)
        restored = dequantize(bundle)
        raw = np.random.default_rng(77).standard_normal(64).astype(np.float32)
        w = _softmax(raw)
        ref_out = w @ restored[0, 1, 0]
        fused_out = attention_output(bundle, w, layer=0, head=0)
        np.testing.assert_allclose(fused_out, ref_out, atol=1e-4)

    def test_all_heads(self):
        kv = _make_kv_cache(n_heads=4)
        bundle = quantize(kv)
        restored = dequantize(bundle)
        raw = np.random.default_rng(77).standard_normal(64).astype(np.float32)
        w = _softmax(raw)
        for head in range(4):
            ref = w @ restored[0, 1, head]
            fused = attention_output(bundle, w, layer=0, head=head)
            np.testing.assert_allclose(fused, ref, atol=1e-4, err_msg=f"head {head}")

    def test_zero_weights(self):
        kv = _make_kv_cache()
        bundle = quantize(kv)
        w = np.zeros(64, dtype=np.float32)
        fused_out = attention_output(bundle, w, layer=0, head=0)
        np.testing.assert_allclose(fused_out, np.zeros(128), atol=1e-7)

    def test_single_position_weight(self):
        kv = _make_kv_cache()
        bundle = quantize(kv)
        restored = dequantize(bundle)
        w = np.zeros(64, dtype=np.float32)
        w[7] = 1.0
        ref = restored[0, 1, 0, 7]
        fused = attention_output(bundle, w, layer=0, head=0)
        np.testing.assert_allclose(fused, ref, atol=1e-4)


class TestAttentionValidation:
    def test_wrong_head_dim(self):
        kv = _make_kv_cache(head_dim=64)
        bundle = quantize(kv)
        q_vec = np.zeros(128, dtype=np.float32)
        with pytest.raises(ValueError, match="head_dim=128"):
            attention_scores(bundle, q_vec, layer=0)

    def test_wrong_query_shape(self):
        kv = _make_kv_cache()
        bundle = quantize(kv)
        q_vec = np.zeros(64, dtype=np.float32)
        with pytest.raises(ValueError, match="shape"):
            attention_scores(bundle, q_vec, layer=0)
