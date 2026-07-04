import sys
import os
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from ops import layer_norm, gelu, softmax


def test_layer_norm_output_has_zero_mean_unit_variance_before_scale_shift():
    x = np.array([[1.0, 2.0, 3.0, 4.0]])
    weight = np.ones(4)
    bias = np.zeros(4)
    out = layer_norm(x, weight, bias)
    assert np.isclose(out.mean(), 0, atol=1e-6)
    assert np.isclose(out.std(), 1, atol=1e-2)


def test_layer_norm_applies_weight_and_bias():
    x = np.array([[1.0, 2.0, 3.0, 4.0]])
    weight = np.array([2.0, 2.0, 2.0, 2.0])
    bias = np.array([1.0, 1.0, 1.0, 1.0])
    out = layer_norm(x, weight, bias)
    baseline = layer_norm(x, np.ones(4), np.zeros(4))
    assert np.allclose(out, baseline * 2.0 + 1.0)


def test_gelu_zero_is_zero():
    assert np.isclose(gelu(np.array([0.0]))[0], 0.0)


def test_gelu_large_positive_approaches_identity():
    x = np.array([10.0])
    assert np.isclose(gelu(x)[0], 10.0, atol=1e-3)


def test_gelu_large_negative_approaches_zero():
    x = np.array([-10.0])
    assert np.isclose(gelu(x)[0], 0.0, atol=1e-3)


def test_softmax_sums_to_one():
    x = np.array([1.0, 2.0, 3.0])
    out = softmax(x)
    assert np.isclose(out.sum(), 1.0)


def test_softmax_is_monotonic():
    x = np.array([1.0, 2.0, 3.0])
    out = softmax(x)
    assert out[0] < out[1] < out[2]


def test_softmax_numerically_stable_for_large_values():
    x = np.array([1000.0, 1001.0, 1002.0])
    out = softmax(x)
    assert not np.any(np.isnan(out))
    assert np.isclose(out.sum(), 1.0)
