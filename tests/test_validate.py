"""Test Q4 validate kernel — checks for NaN/Inf/negative scale detection."""

import numpy as np
import pytest

from eakv._ops import q4_validate


class TestValidate:
    def test_valid_data(self):
        """Valid scales and biases should return 0."""
        scales = np.array([1.0, 2.0, 0.5, 0.0], dtype=np.float32)
        biases = np.array([0.0, -1.0, 3.5, 0.0], dtype=np.float32)
        assert q4_validate(scales, biases, 4) == 0

    def test_nan_scale(self):
        """NaN in scales should return error code 1."""
        scales = np.array([1.0, float("nan"), 0.5], dtype=np.float32)
        biases = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        assert q4_validate(scales, biases, 3) == 1

    def test_nan_bias(self):
        """NaN in biases should return error code 2."""
        scales = np.array([1.0, 2.0, 0.5], dtype=np.float32)
        biases = np.array([0.0, float("nan"), 0.0], dtype=np.float32)
        assert q4_validate(scales, biases, 3) == 2

    def test_negative_scale(self):
        """Negative scale should return error code 3."""
        scales = np.array([1.0, -0.5, 0.5], dtype=np.float32)
        biases = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        assert q4_validate(scales, biases, 3) == 3

    def test_empty(self):
        """Zero groups should return 0."""
        scales = np.array([], dtype=np.float32)
        biases = np.array([], dtype=np.float32)
        assert q4_validate(scales, biases, 0) == 0

    def test_nan_scale_first_group(self):
        """NaN in first group scale should be caught immediately."""
        scales = np.array([float("nan")], dtype=np.float32)
        biases = np.array([0.0], dtype=np.float32)
        assert q4_validate(scales, biases, 1) == 1

    def test_inf_scale(self):
        """Inf is not NaN and is non-negative, so should pass validation."""
        scales = np.array([float("inf")], dtype=np.float32)
        biases = np.array([0.0], dtype=np.float32)
        assert q4_validate(scales, biases, 1) == 0

    def test_neg_inf_scale(self):
        """Negative inf should return error code 3 (negative scale)."""
        scales = np.array([float("-inf")], dtype=np.float32)
        biases = np.array([0.0], dtype=np.float32)
        assert q4_validate(scales, biases, 1) == 3

    def test_inf_bias(self):
        """Inf bias is not NaN, so should pass validation."""
        scales = np.array([1.0], dtype=np.float32)
        biases = np.array([float("inf")], dtype=np.float32)
        assert q4_validate(scales, biases, 1) == 0

    def test_priority_nan_scale_over_negative(self):
        """NaN scale (code 1) should be returned before negative scale (code 3)."""
        scales = np.array([float("nan")], dtype=np.float32)
        biases = np.array([0.0], dtype=np.float32)
        # NaN is checked before negative, so code 1
        assert q4_validate(scales, biases, 1) == 1
