import sys
import os
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))

from gpt2_model import GPT2Model, GPT2ModelCached
from weight_loader import GPT2Weights
from kv_cache import KVCache

MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models", "gpt2-124m")


def make_weights():
    return GPT2Weights(MODEL_DIR)


def test_cached_generation_matches_full_recompute_logits():
    weights = make_weights()
    plain_model = GPT2Model(weights)
    cached_model = GPT2ModelCached(weights)

    token_ids = [464, 2003, 286, 11666, 4430, 318]

    full_logits = plain_model.forward(token_ids)

    cache = KVCache(weights.n_layer)
    cached_logits_prompt = cached_model.forward_with_cache(token_ids[:4], cache)
    cached_logits_next = cached_model.forward_with_cache(token_ids[4:5], cache)
    cached_logits_last = cached_model.forward_with_cache(token_ids[5:6], cache)

    assert np.allclose(full_logits[3], cached_logits_prompt[-1], atol=1e-4)
    assert np.allclose(full_logits[4], cached_logits_next[-1], atol=1e-4)
    assert np.allclose(full_logits[5], cached_logits_last[-1], atol=1e-4)


def test_kv_cache_position_tracking():
    weights = make_weights()
    cache = KVCache(weights.n_layer)
    model = GPT2ModelCached(weights)

    assert cache.position == 0
    model.forward_with_cache([1, 2, 3], cache)
    assert cache.position == 3
    model.forward_with_cache([4], cache)
    assert cache.position == 4


def test_cache_grows_correctly_across_multiple_steps():
    weights = make_weights()
    cache = KVCache(weights.n_layer)
    model = GPT2ModelCached(weights)

    model.forward_with_cache([10, 20, 30], cache)
    assert cache.k[0].shape[1] == 3

    model.forward_with_cache([40], cache)
    assert cache.k[0].shape[1] == 4
