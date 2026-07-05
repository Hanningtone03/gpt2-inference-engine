import numpy as np


def generate(model, tokenizer, prompt, max_new_tokens=30, strategy="greedy",
             temperature=1.0, top_k=40, top_p=0.9, seed=None):
    from sampling import greedy_sample, temperature_sample, top_k_sample, top_p_sample

    rng = np.random.default_rng(seed)
    token_ids = tokenizer.encode(prompt)

    for _ in range(max_new_tokens):
        logits = model.forward(token_ids)
        next_token_logits = logits[-1]

        if strategy == "greedy":
            next_id = greedy_sample(next_token_logits)
        elif strategy == "temperature":
            next_id = temperature_sample(next_token_logits, temperature, rng)
        elif strategy == "top_k":
            next_id = top_k_sample(next_token_logits, top_k, temperature, rng)
        elif strategy == "top_p":
            next_id = top_p_sample(next_token_logits, top_p, temperature, rng)
        else:
            raise ValueError(f"unknown sampling strategy: {strategy}")

        token_ids.append(next_id)

        if next_id == tokenizer.encoder.get("<|endoftext|>"):
            break

    return tokenizer.decode(token_ids)


def generate_cached(model_cached, tokenizer, prompt, max_new_tokens=30, strategy="greedy",
                     temperature=1.0, top_k=40, top_p=0.9, seed=None):
    from sampling import greedy_sample, temperature_sample, top_k_sample, top_p_sample
    from kv_cache import KVCache

    rng = np.random.default_rng(seed)
    token_ids = tokenizer.encode(prompt)
    cache = KVCache(model_cached.weights.n_layer)

    logits = model_cached.forward_with_cache(token_ids, cache)
    next_token_logits = logits[-1]

    for _ in range(max_new_tokens):
        if strategy == "greedy":
            next_id = greedy_sample(next_token_logits)
        elif strategy == "temperature":
            next_id = temperature_sample(next_token_logits, temperature, rng)
        elif strategy == "top_k":
            next_id = top_k_sample(next_token_logits, top_k, temperature, rng)
        elif strategy == "top_p":
            next_id = top_p_sample(next_token_logits, top_p, temperature, rng)
        else:
            raise ValueError(f"unknown sampling strategy: {strategy}")

        token_ids.append(next_id)

        if next_id == tokenizer.encoder.get("<|endoftext|>"):
            break

        logits = model_cached.forward_with_cache([next_id], cache)
        next_token_logits = logits[-1]

    return tokenizer.decode(token_ids)
