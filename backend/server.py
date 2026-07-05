import json
import sys
import os
import time

sys.path.insert(0, os.path.dirname(__file__))

from flask import Flask, request, Response
from flask_cors import CORS
import numpy as np

from gpt2_model import GPT2ModelCachedWithAttention
from weight_loader import GPT2Weights
from bpe_tokenizer import BPETokenizer
from kv_cache import KVCache
from sampling import greedy_sample, temperature_sample, top_k_sample, top_p_sample
from ops import softmax

MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models", "gpt2-124m")
MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")

app = Flask(__name__)
CORS(app)

print("loading weights...")
weights = GPT2Weights(MODEL_DIR)
model = GPT2ModelCachedWithAttention(weights)
tokenizer = BPETokenizer(
    os.path.join(MODELS_DIR, "encoder.json"),
    os.path.join(MODELS_DIR, "vocab.bpe"),
)
print("model ready")


def sample_next(logits, strategy, temperature, top_k, top_p, rng):
    if strategy == "greedy":
        return greedy_sample(logits)
    if strategy == "temperature":
        return temperature_sample(logits, temperature, rng)
    if strategy == "top_k":
        return top_k_sample(logits, top_k, temperature, rng)
    if strategy == "top_p":
        return top_p_sample(logits, top_p, temperature, rng)
    raise ValueError(f"unknown strategy: {strategy}")


def top_candidates(logits, n=5):
    probs = softmax(logits)
    top_indices = np.argsort(probs)[-n:][::-1]
    return [
        {"text": tokenizer.decode([int(idx)]), "prob": float(probs[idx])}
        for idx in top_indices
    ]


@app.route("/api/tokenize", methods=["POST"])
def tokenize():
    data = request.get_json()
    text = data.get("text", "")
    pieces = tokenizer.encode_with_pieces(text)
    return {"pieces": pieces}


@app.route("/api/chat", methods=["POST"])
def chat():
    data = request.get_json()
    prompt = data.get("prompt", "")
    max_new_tokens = min(int(data.get("max_new_tokens", 60)), 200)
    strategy = data.get("strategy", "top_k")
    temperature = float(data.get("temperature", 0.8))
    top_k = int(data.get("top_k", 40))
    top_p = float(data.get("top_p", 0.9))
    seed = data.get("seed")

    def stream():
        rng = np.random.default_rng(seed)
        token_ids = tokenizer.encode(prompt)
        prompt_len = len(token_ids)
        cache = KVCache(model.weights.n_layer)

        start_time = time.time()
        tokens_generated = 0

        logits, attn_weights = model.forward_with_cache_and_attention(token_ids, cache)
        next_logits = logits[-1]

        prompt_attention = attn_weights[-1].tolist()
        prompt_pieces = tokenizer.encode_with_pieces(prompt)
        yield f"data: {json.dumps({'promptPieces': prompt_pieces, 'promptAttention': prompt_attention})}\n\n"

        for _ in range(max_new_tokens):
            candidates = top_candidates(next_logits, n=5)
            probs = softmax(next_logits)

            next_id = sample_next(next_logits, strategy, temperature, top_k, top_p, rng)
            chosen_prob = float(probs[next_id])

            if next_id == tokenizer.encoder.get("<|endoftext|>"):
                break

            piece = tokenizer.decode([next_id])
            tokens_generated += 1
            elapsed = time.time() - start_time
            tps = tokens_generated / elapsed if elapsed > 0 else 0

            yield f"data: {json.dumps({'token': piece, 'prob': chosen_prob, 'candidates': candidates, 'tokensPerSec': tps, 'elapsed': elapsed})}\n\n"

            token_ids.append(next_id)
            logits, attn_weights = model.forward_with_cache_and_attention([next_id], cache)
            next_logits = logits[-1]

        yield f"data: {json.dumps({'done': True})}\n\n"

    return Response(stream(), mimetype="text/event-stream")


@app.route("/api/compare", methods=["POST"])
def compare():
    data = request.get_json()
    prompt = data.get("prompt", "")
    max_new_tokens = min(int(data.get("max_new_tokens", 40)), 100)
    strategy_a = data.get("strategyA", "greedy")
    strategy_b = data.get("strategyB", "top_k")
    seed = data.get("seed", 42)

    def run_strategy(strategy):
        rng = np.random.default_rng(seed)
        token_ids = tokenizer.encode(prompt)
        cache = KVCache(model.weights.n_layer)
        logits, _ = model.forward_with_cache_and_attention(token_ids, cache)
        next_logits = logits[-1]

        for _ in range(max_new_tokens):
            next_id = sample_next(next_logits, strategy, 0.8, 40, 0.9, rng)
            if next_id == tokenizer.encoder.get("<|endoftext|>"):
                break
            token_ids.append(next_id)
            logits, _ = model.forward_with_cache_and_attention([next_id], cache)
            next_logits = logits[-1]

        return tokenizer.decode(token_ids)

    result_a = run_strategy(strategy_a)
    result_b = run_strategy(strategy_b)

    return {"resultA": result_a, "resultB": result_b}


@app.route("/api/health", methods=["GET"])
def health():
    return {"status": "ok"}


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    app.run(host="0.0.0.0", port=port, threaded=True)
