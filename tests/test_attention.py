"""Test fused attention kernels against reference restore-then-compute."""

import numpy as np
import pytest

from eakv._quantize import quantize
from eakv._restore import dequantize
from eakv._attention import (
    attention_scores, attention_output,
    attention_scores_multi, attention_output_multi,
    attention_fused, attention_fused_multi,
)


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


class TestFusedKScoreMulti:
    def test_matches_per_head(self):
        """Multi-head kernel must match per-head kernel for all heads."""
        kv = _make_kv_cache(n_heads=4)
        bundle = quantize(kv)
        q_vec = np.random.default_rng(99).standard_normal(128).astype(np.float32)
        q_vecs = np.tile(q_vec, (4, 1))  # same query for all heads
        multi_scores = attention_scores_multi(bundle, q_vecs, layer=0, n_heads=4)
        for head in range(4):
            single = attention_scores(bundle, q_vec, layer=0, head=head)
            np.testing.assert_allclose(
                multi_scores[head], single, atol=1e-5, err_msg=f"head {head}"
            )

    def test_different_queries(self):
        """Each head gets its own query vector."""
        kv = _make_kv_cache(n_heads=4)
        bundle = quantize(kv)
        restored = dequantize(bundle)
        rng = np.random.default_rng(123)
        q_vecs = rng.standard_normal((4, 128)).astype(np.float32)
        multi_scores = attention_scores_multi(bundle, q_vecs, layer=0, n_heads=4)
        for head in range(4):
            ref = restored[0, 0, head] @ q_vecs[head] / np.sqrt(128.0)
            np.testing.assert_allclose(
                multi_scores[head], ref, atol=1e-4, err_msg=f"head {head}"
            )


class TestFusedVSumMulti:
    def test_matches_per_head(self):
        """Multi-head kernel must match per-head kernel for all heads."""
        kv = _make_kv_cache(n_heads=4)
        bundle = quantize(kv)
        raw = np.random.default_rng(77).standard_normal(64).astype(np.float32)
        w = _softmax(raw)
        all_w = np.tile(w, (4, 1))
        multi_out = attention_output_multi(bundle, all_w, layer=0, n_heads=4)
        for head in range(4):
            single = attention_output(bundle, w, layer=0, head=head)
            np.testing.assert_allclose(
                multi_out[head], single, atol=1e-5, err_msg=f"head {head}"
            )

    def test_different_weights(self):
        """Each head gets its own weight vector."""
        kv = _make_kv_cache(n_heads=4)
        bundle = quantize(kv)
        restored = dequantize(bundle)
        rng = np.random.default_rng(55)
        all_w = np.zeros((4, 64), dtype=np.float32)
        for h in range(4):
            all_w[h] = _softmax(rng.standard_normal(64).astype(np.float32))
        multi_out = attention_output_multi(bundle, all_w, layer=0, n_heads=4)
        for head in range(4):
            ref = all_w[head] @ restored[0, 1, head]
            np.testing.assert_allclose(
                multi_out[head], ref, atol=1e-4, err_msg=f"head {head}"
            )


class TestFusedAttention:
    def test_matches_separate_kernels(self):
        """Fused attention must match scores→softmax→output pipeline."""
        kv = _make_kv_cache()
        bundle = quantize(kv)
        q_vec = np.random.default_rng(99).standard_normal(128).astype(np.float32)
        # Reference: separate kernels
        scores = attention_scores(bundle, q_vec, layer=0, head=0)
        weights = _softmax(scores)
        ref_out = attention_output(bundle, weights, layer=0, head=0)
        # Fused
        fused_out = attention_fused(bundle, q_vec, layer=0, head=0)
        np.testing.assert_allclose(fused_out, ref_out, atol=1e-4)

    def test_all_heads(self):
        kv = _make_kv_cache(n_heads=4)
        bundle = quantize(kv)
        q_vec = np.random.default_rng(99).standard_normal(128).astype(np.float32)
        for head in range(4):
            scores = attention_scores(bundle, q_vec, layer=0, head=head)
            weights = _softmax(scores)
            ref = attention_output(bundle, weights, layer=0, head=head)
            fused = attention_fused(bundle, q_vec, layer=0, head=head)
            np.testing.assert_allclose(fused, ref, atol=1e-4, err_msg=f"head {head}")

    def test_matches_numpy_reference(self):
        """Fused kernel must match full numpy softmax(QK^T/sqrt(d))·V."""
        kv = _make_kv_cache()
        bundle = quantize(kv)
        restored = dequantize(bundle)
        q_vec = np.random.default_rng(99).standard_normal(128).astype(np.float32)
        k_head = restored[0, 0, 0]
        v_head = restored[0, 1, 0]
        scores = k_head @ q_vec / np.sqrt(128.0)
        ref_out = _softmax(scores) @ v_head
        fused_out = attention_fused(bundle, q_vec, layer=0, head=0)
        np.testing.assert_allclose(fused_out, ref_out, atol=1e-3)

    def test_longer_sequence(self):
        kv = _make_kv_cache(seq_len=512)
        bundle = quantize(kv)
        q_vec = np.random.default_rng(99).standard_normal(128).astype(np.float32)
        scores = attention_scores(bundle, q_vec, layer=0, head=0)
        weights = _softmax(scores)
        ref = attention_output(bundle, weights, layer=0, head=0)
        fused = attention_fused(bundle, q_vec, layer=0, head=0)
        np.testing.assert_allclose(fused, ref, atol=1e-4)


class TestFusedAttentionMulti:
    def test_matches_per_head(self):
        kv = _make_kv_cache(n_heads=4)
        bundle = quantize(kv)
        q_vecs = np.random.default_rng(123).standard_normal((4, 128)).astype(np.float32)
        multi_out = attention_fused_multi(bundle, q_vecs, layer=0, n_heads=4)
        for head in range(4):
            single = attention_fused(bundle, q_vecs[head], layer=0, head=head)
            np.testing.assert_allclose(
                multi_out[head], single, atol=1e-5, err_msg=f"head {head}"
            )

    def test_matches_numpy_reference(self):
        kv = _make_kv_cache(n_heads=4)
        bundle = quantize(kv)
        restored = dequantize(bundle)
        q_vecs = np.random.default_rng(123).standard_normal((4, 128)).astype(np.float32)
        multi_out = attention_fused_multi(bundle, q_vecs, layer=0, n_heads=4)
        for head in range(4):
            k_head = restored[0, 0, head]
            v_head = restored[0, 1, head]
            scores = k_head @ q_vecs[head] / np.sqrt(128.0)
            ref = _softmax(scores) @ v_head
            np.testing.assert_allclose(
                multi_out[head], ref, atol=1e-3, err_msg=f"head {head}"
            )


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
