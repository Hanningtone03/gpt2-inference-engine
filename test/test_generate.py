import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))

from generate import generate
from gpt2_model import GPT2Model
from weight_loader import GPT2Weights
from bpe_tokenizer import BPETokenizer

MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models", "gpt2-124m")
MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")


def make_model():
    weights = GPT2Weights(MODEL_DIR)
    return GPT2Model(weights)


def make_tokenizer():
    return BPETokenizer(
        os.path.join(MODELS_DIR, "encoder.json"),
        os.path.join(MODELS_DIR, "vocab.bpe"),
    )


def test_greedy_generation_produces_longer_text_than_prompt():
    model = make_model()
    tokenizer = make_tokenizer()
    prompt = "The weather today is"
    result = generate(model, tokenizer, prompt, max_new_tokens=10, strategy="greedy")
    assert len(result) > len(prompt)
    assert result.startswith(prompt)


def test_greedy_generation_is_deterministic():
    model = make_model()
    tokenizer = make_tokenizer()
    prompt = "Once upon a time"
    result1 = generate(model, tokenizer, prompt, max_new_tokens=8, strategy="greedy")
    result2 = generate(model, tokenizer, prompt, max_new_tokens=8, strategy="greedy")
    assert result1 == result2


def test_seeded_top_k_generation_is_reproducible():
    model = make_model()
    tokenizer = make_tokenizer()
    prompt = "In a distant galaxy"
    result1 = generate(model, tokenizer, prompt, max_new_tokens=8, strategy="top_k", seed=123)
    result2 = generate(model, tokenizer, prompt, max_new_tokens=8, strategy="top_k", seed=123)
    assert result1 == result2
