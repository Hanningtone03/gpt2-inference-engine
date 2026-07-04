import numpy as np


def layer_norm(x, weight, bias, eps=1e-5):
    mean = x.mean(axis=-1, keepdims=True)
    variance = x.var(axis=-1, keepdims=True)
    normalized = (x - mean) / np.sqrt(variance + eps)
    return normalized * weight + bias


def gelu(x):
    return 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * x ** 3)))


def softmax(x, axis=-1):
    shifted = x - np.max(x, axis=axis, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.sum(exp, axis=axis, keepdims=True)
