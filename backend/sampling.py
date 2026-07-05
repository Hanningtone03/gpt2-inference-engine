import numpy as np
from ops import softmax


def greedy_sample(logits):
    return int(np.argmax(logits))


def temperature_sample(logits, temperature, rng):
    scaled = logits / temperature
    probs = softmax(scaled)
    return int(rng.choice(len(probs), p=probs))


def top_k_sample(logits, k, temperature, rng):
    top_k_indices = np.argsort(logits)[-k:]
    top_k_logits = logits[top_k_indices]
    scaled = top_k_logits / temperature
    probs = softmax(scaled)
    chosen_idx = rng.choice(len(probs), p=probs)
    return int(top_k_indices[chosen_idx])


def top_p_sample(logits, p, temperature, rng):
    scaled = logits / temperature
    probs = softmax(scaled)

    sorted_indices = np.argsort(probs)[::-1]
    sorted_probs = probs[sorted_indices]
    cumulative = np.cumsum(sorted_probs)

    cutoff = np.searchsorted(cumulative, p) + 1
    cutoff = max(cutoff, 1)

    kept_indices = sorted_indices[:cutoff]
    kept_probs = sorted_probs[:cutoff]
    kept_probs = kept_probs / kept_probs.sum()

    chosen_idx = rng.choice(len(kept_probs), p=kept_probs)
    return int(kept_indices[chosen_idx])
