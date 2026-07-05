import sys
import os
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))

from gpt2_model import GPT2ModelCachedWithAttention
from weight_loader import GPT2Weights
from kv_cache import KVCache

MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models", "gpt2-124m")


def test_attention_weights_shape_and_sum_to_one():
    weights = GPT2Weights(MODEL_DIR)
    model = GPT2ModelCachedWithAttention(weights)
    cache = KVCache(weights.n_layer)

    token_ids = [464, 3290, 318, 257]
    logits, attn_weights = model.forward_with_cache_and_attention(token_ids, cache)

    assert attn_weights.shape == (4, 4)
    row_sums = attn_weights.sum(axis=-1)
    assert np.allclose(row_sums, 1.0, atol=1e-4)


def test_attention_weights_respect_causal_mask():
    weights = GPT2Weights(MODEL_DIR)
    model = GPT2ModelCachedWithAttention(weights)
    cache = KVCache(weights.n_layer)

    token_ids = [464, 3290, 318]
    logits, attn_weights = model.forward_with_cache_and_attention(token_ids, cache)

    assert np.allclose(attn_weights[0, 1:], 0.0, atol=1e-6)
    assert np.allclose(attn_weights[1, 2:], 0.0, atol=1e-6)
