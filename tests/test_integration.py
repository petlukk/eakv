"""End-to-end integration tests for eakv."""

import tempfile
import numpy as np
import eakv


def test_full_workflow():
    rng = np.random.default_rng(42)
    kv_cache = rng.standard_normal((8, 2, 4, 128, 64)).astype(np.float32)

    bundle = eakv.quantize(kv_cache)
    assert bundle.n_layers == 8
    assert bundle.compression_ratio < 1.0

    eakv.validate(bundle)

    with tempfile.NamedTemporaryFile(suffix=".eakv") as f:
        eakv.save(bundle, f.name)
        loaded = eakv.load(f.name)
        full = eakv.dequantize(loaded)
        assert full.shape == kv_cache.shape
        max_err = np.max(np.abs(full - kv_cache))
        assert max_err < 0.5

        with eakv.open_mmap(f.name) as mmapped:
            partial = eakv.restore(mmapped, layers=range(4), tokens=-64)
            assert partial.shape == (4, 2, 4, 64, 64)

            single = eakv.restore(mmapped, layers=0, heads=0)
            assert single.shape == (1, 2, 1, 128, 64)


def test_compression_ratio():
    kv = np.random.randn(32, 2, 32, 128, 128).astype(np.float32)
    bundle = eakv.quantize(kv)
    # weights are u8 (1 byte each, 32 per group = 32 bytes) + 4 scale + 4 bias = 40 bytes per group
    # original: 64 * 4 = 256 bytes per group (f32)
    # ratio: 40/256 ~ 0.156
    assert bundle.compression_ratio < 0.25
    assert bundle.compression_ratio > 0.1
