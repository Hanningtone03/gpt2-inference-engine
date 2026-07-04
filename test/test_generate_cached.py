import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from generate import generate, generate_cached
from gpt2_model import GPT2Model, GPT2ModelCached
from weight_loader import GPT2Weights
from bpe_tokenizer import BPETokenizer

MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models", "gpt2-124m")
MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")


def make_tokenizer():
    return BPETokenizer(
        os.path.join(MODELS_DIR, "encoder.json"),
        os.path.join(MODELS_DIR, "vocab.bpe"),
    )


def test_cached_generation_matches_uncached_output_for_greedy():
    weights = GPT2Weights(MODEL_DIR)
    plain_model = GPT2Model(weights)
    cached_model = GPT2ModelCached(weights)
    tokenizer = make_tokenizer()

    prompt = "The scientist discovered"
    uncached_result = generate(plain_model, tokenizer, prompt, max_new_tokens=10, strategy="greedy")
    cached_result = generate_cached(cached_model, tokenizer, prompt, max_new_tokens=10, strategy="greedy")

    assert uncached_result == cached_result
