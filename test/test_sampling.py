import sys
import os
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))

from sampling import greedy_sample, temperature_sample, top_k_sample, top_p_sample


def test_greedy_sample_picks_highest_logit():
    logits = np.array([1.0, 5.0, 2.0, 0.5])
    assert greedy_sample(logits) == 1


def test_temperature_sample_is_deterministic_with_seeded_rng():
    logits = np.array([1.0, 2.0, 3.0, 4.0])
    rng1 = np.random.default_rng(42)
    rng2 = np.random.default_rng(42)
    assert temperature_sample(logits, 1.0, rng1) == temperature_sample(logits, 1.0, rng2)


def test_top_k_sample_only_returns_indices_in_top_k():
    logits = np.array([0.1, 5.0, 0.2, 4.5, 0.3, 4.0, 0.1, 0.1])
    rng = np.random.default_rng(0)
    top_3_expected = set(np.argsort(logits)[-3:].tolist())

    for _ in range(20):
        result = top_k_sample(logits, k=3, temperature=1.0, rng=rng)
        assert result in top_3_expected


def test_top_p_sample_returns_valid_index():
    logits = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    rng = np.random.default_rng(0)
    result = top_p_sample(logits, p=0.9, temperature=1.0, rng=rng)
    assert 0 <= result < len(logits)


def test_top_p_with_very_low_p_behaves_like_greedy():
    logits = np.array([1.0, 2.0, 10.0, 3.0])
    rng = np.random.default_rng(0)
    result = top_p_sample(logits, p=0.01, temperature=1.0, rng=rng)
    assert result == 2
