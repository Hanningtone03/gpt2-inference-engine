import sys
import os
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from attention import causal_self_attention, split_heads, merge_heads


def test_split_and_merge_heads_are_inverses():
    x = np.random.randn(5, 12)
    split = split_heads(x, n_head=3)
    merged = merge_heads(split)
    assert np.allclose(x, merged)


def test_split_heads_shape():
    x = np.random.randn(5, 12)
    split = split_heads(x, n_head=3)
    assert split.shape == (3, 5, 4)


def test_attention_output_shape_matches_input():
    seq_len, n_embd, n_head = 6, 12, 3
    x = np.random.randn(seq_len, n_embd)
    c_attn_weight = np.random.randn(n_embd, n_embd * 3) * 0.02
    c_attn_bias = np.zeros(n_embd * 3)
    c_proj_weight = np.random.randn(n_embd, n_embd) * 0.02
    c_proj_bias = np.zeros(n_embd)

    out = causal_self_attention(x, c_attn_weight, c_attn_bias, c_proj_weight, c_proj_bias, n_head)
    assert out.shape == (seq_len, n_embd)


def test_causal_masking_first_token_only_attends_to_itself():
    seq_len, n_embd, n_head = 4, 8, 2
    np.random.seed(0)
    x = np.random.randn(seq_len, n_embd)
    c_attn_weight = np.random.randn(n_embd, n_embd * 3) * 0.02
    c_attn_bias = np.zeros(n_embd * 3)

    qkv = x @ c_attn_weight + c_attn_bias
    q, k, v = np.split(qkv, 3, axis=-1)
    q_heads = split_heads(q, n_head)
    k_heads = split_heads(k, n_head)

    head_dim = n_embd // n_head
    scores = (q_heads @ k_heads.transpose(0, 2, 1)) / np.sqrt(head_dim)
    causal_mask = np.triu(np.ones((seq_len, seq_len)), k=1).astype(bool)
    masked_scores = np.where(causal_mask, -np.inf, scores)

    from ops import softmax
    weights = softmax(masked_scores, axis=-1)

    assert np.allclose(weights[:, 0, 1:], 0.0)
    assert np.isclose(weights[:, 0, 0], 1.0).all()


def test_changing_a_future_token_does_not_affect_earlier_token_output():
    seq_len, n_embd, n_head = 5, 8, 2
    np.random.seed(1)
    c_attn_weight = np.random.randn(n_embd, n_embd * 3) * 0.02
    c_attn_bias = np.zeros(n_embd * 3)
    c_proj_weight = np.random.randn(n_embd, n_embd) * 0.02
    c_proj_bias = np.zeros(n_embd)

    x1 = np.random.randn(seq_len, n_embd)
    x2 = x1.copy()
    x2[-1] = np.random.randn(n_embd)

    out1 = causal_self_attention(x1, c_attn_weight, c_attn_bias, c_proj_weight, c_proj_bias, n_head)
    out2 = causal_self_attention(x2, c_attn_weight, c_attn_bias, c_proj_weight, c_proj_bias, n_head)

    assert np.allclose(out1[:-1], out2[:-1])
