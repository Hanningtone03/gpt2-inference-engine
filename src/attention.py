import numpy as np
from ops import softmax


def split_heads(x, n_head):
    seq_len, n_embd = x.shape
    head_dim = n_embd // n_head
    x = x.reshape(seq_len, n_head, head_dim)
    return x.transpose(1, 0, 2)


def merge_heads(x):
    n_head, seq_len, head_dim = x.shape
    x = x.transpose(1, 0, 2)
    return x.reshape(seq_len, n_head * head_dim)


def causal_self_attention(x, c_attn_weight, c_attn_bias, c_proj_weight, c_proj_bias, n_head):
    seq_len, n_embd = x.shape

    qkv = x @ c_attn_weight + c_attn_bias
    q, k, v = np.split(qkv, 3, axis=-1)

    q = split_heads(q, n_head)
    k = split_heads(k, n_head)
    v = split_heads(v, n_head)

    head_dim = n_embd // n_head
    scores = (q @ k.transpose(0, 2, 1)) / np.sqrt(head_dim)

    causal_mask = np.triu(np.ones((seq_len, seq_len)), k=1).astype(bool)
    scores = np.where(causal_mask, -np.inf, scores)

    weights = softmax(scores, axis=-1)
    attn_out = weights @ v

    merged = merge_heads(attn_out)
    return merged @ c_proj_weight + c_proj_bias
